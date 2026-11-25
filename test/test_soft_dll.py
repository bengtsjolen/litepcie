#
# This file is part of LitePCIe.
#
# Copyright (c) 2026 Florent Kermarrec <florent@enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import random
import unittest

import pytest

from migen import *

from litepcie.phy.soft.common   import STP, SDP
from litepcie.phy.soft.dll.core import DataLinkLayer, rx_buffer_size

from test.model.soft.crc import dllp_crc_bytes

pytestmark = pytest.mark.slow # About 2 minutes with the Migen simulator.

class DLLLink(Module):
    """Two Data Link Layers a and b connected by ideal lanes (Logical Idle filler, no word phase
    change), with error injection: corrupt_a2b (pulse) flips a bit in the first DW of the next TLP
    from a to b, drop_dllp_b2a (level) drops the DLLPs from b to a (Acks lost)."""
    def __init__(self):
        self.submodules.a = a = DataLinkLayer(125e6, rx_depth=128, tx_depth=128, descriptors=8,
            replay_timer=256, update_timer=1500, with_rx_merge=True)
        self.submodules.b = b = DataLinkLayer(125e6, rx_depth=128, tx_depth=128, descriptors=8,
            replay_timer=256, update_timer=1500, with_rx_merge=True)
        self.link_up      = Signal()
        self.corrupt_a2b  = Signal()
        self.drop_dllp_b2a = Signal()
        self.comb += [
            a.link_up.eq(self.link_up), a.l0.eq(self.link_up),
            b.link_up.eq(self.link_up), b.l0.eq(self.link_up),
        ]
        # a -> b with corruption.
        arm         = Signal()
        corrupt_now = Signal()
        is_stp      = a.phy_sink.valid & a.phy_sink.ctrl[0] & (a.phy_sink.data[0:8] == STP.value)
        self.sync += [
            corrupt_now.eq(0),
            If(self.corrupt_a2b, arm.eq(1)),
            If(arm & is_stp,
                arm.eq(0),
                corrupt_now.eq(1),
            )
        ]
        self.comb += [
            a.phy_sink.ready.eq(1),
            b.phy_source.valid.eq(1),
            If(a.phy_sink.valid,
                b.phy_source.data.eq(a.phy_sink.data ^ (corrupt_now << 13)),
                b.phy_source.ctrl.eq(a.phy_sink.ctrl),
            )
        ]
        # b -> a with DLLP drops (SDP word and the next one).
        dropping = Signal()
        is_sdp   = b.phy_sink.valid & b.phy_sink.ctrl[0] & (b.phy_sink.data[0:8] == SDP.value)
        drop     = Signal()
        self.comb += drop.eq(self.drop_dllp_b2a & (is_sdp | dropping))
        self.sync += If(b.phy_sink.valid, dropping.eq(self.drop_dllp_b2a & is_sdp))
        self.comb += [
            b.phy_sink.ready.eq(1),
            a.phy_source.valid.eq(1),
            If(b.phy_sink.valid & ~drop,
                a.phy_source.data.eq(b.phy_sink.data),
                a.phy_source.ctrl.eq(b.phy_sink.ctrl),
            )
        ]

def mwr_tlp(rng, ndws, tag):
    """Memory Write TLP DW words (byte 0 on bits 7:0): 3DW header + ndws DWs."""
    header = bytes([0x40, 0x00, (ndws >> 8) & 0x3, ndws & 0xff])       # Fmt 010 Type 00000, Length.
    header += bytes([0x01, 0x00, tag, 0x0f])                            # Requester ID, Tag, BE.
    header += bytes([0x00, 0x00, 0x10, 0x00])                           # Address.
    payload = bytes(rng.getrandbits(8) for _ in range(4*ndws))
    data = header + payload
    return [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]

def cpl_tlp(tag):
    """Completion without data (Cpl) TLP DW words: 3DW header, Type 0b01010 (Completion credit)."""
    data  = bytes([0x0a, 0x00, 0x00, 0x00])  # Fmt 000 Type 01010 (Cpl), Length 0.
    data += bytes([0x01, 0x00, 0x00, 0x00])  # Completer ID, Status, Byte Count.
    data += bytes([0x02, 0x00, tag,  0x00])  # Requester ID, Tag, Lower Address.
    return [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]

class WideDLLLink(Module):
    """Two wide Data Link Layers a and b (beats of ndw DWs, 8b/10b framing on ndw lanes) connected
    by ideal lanes, with the default credits."""
    def __init__(self, ndw):
        kwargs = dict(ndw=ndw, nlanes=ndw, rx_depth=64*ndw, tx_depth=64*ndw, descriptors=16,
            replay_timer=600, update_timer=1500, with_rx_merge=True)
        self.submodules.a = a = DataLinkLayer(125e6, **kwargs)
        self.submodules.b = b = DataLinkLayer(125e6, **kwargs)
        self.link_up = Signal()
        self.comb += [
            a.link_up.eq(self.link_up), a.l0.eq(self.link_up),
            b.link_up.eq(self.link_up), b.l0.eq(self.link_up),
            a.phy_sink.ready.eq(1),
            b.phy_sink.ready.eq(1),
            a.phy_source.valid.eq(1),
            b.phy_source.valid.eq(1),
            If(a.phy_sink.valid,
                b.phy_source.data.eq(a.phy_sink.data),
                b.phy_source.ctrl.eq(a.phy_sink.ctrl),
            ),
            If(b.phy_sink.valid,
                a.phy_source.data.eq(b.phy_sink.data),
                a.phy_source.ctrl.eq(b.phy_sink.ctrl),
            ),
        ]

def np_tlp(byte0, ndws, length, tag):
    """Non-Posted TLP DW words: ndws DWs, byte0 (Fmt/Type) and Length as given. The Data Link Layer
    only classifies it (flow-control type and data credits) from Fmt, Type and Length."""
    data  = bytes([byte0, 0x00, (length >> 8) & 0x3, length & 0xff])
    data += bytes([0x01, 0x00, tag, 0x0f])
    data += bytes((tag + i) & 0xff for i in range(4*(ndws - 2)))
    return [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]

def np_wide_tlp(tag):
    """4 DW header + 5 DW payload (AtomicOp Swap encoding): 1 header and 2 data credits, 9 DWs."""
    return np_tlp(0x6d, 9, 5, tag)

def np_cfgwr_tlp(tag):
    """CfgWr0, 3 DW header + 1 DW payload: 1 header and 1 data credit, 4 DWs."""
    return np_tlp(0x44, 4, 1, tag)

def np_mrd64_tlp(tag):
    """MRd64, 4 DW header: 1 header credit, no data credit, 4 DWs."""
    return np_tlp(0x20, 4, 1, tag)

class TestSoftDLL(unittest.TestCase):
    def test_rx_buffer_size(self):
        # Depth in DWs of storage: the credits' content, plus a partial last beat per header credit.
        self.assertEqual(rx_buffer_size(8, 8, ndw=1),    (64, 8))
        self.assertEqual(rx_buffer_size(16, 128, ndw=1), (1024, 16))
        for ndw in [2, 4, 8, 16]:
            for hdr, data in [(8, 8), (16, 128), (1, 0), (0, 1), (4, 60)]:
                depth, packets = rx_buffer_size(hdr, data, ndw=ndw)
                self.assertEqual(depth % ndw, 0)
                self.assertGreaterEqual(depth, 4*hdr + 4*data + hdr*(ndw - 1))
                self.assertGreaterEqual(packets, hdr)
        self.assertGreaterEqual(rx_buffer_size(8, 8, ndw=8)[0], 128)
        with self.assertRaises(AssertionError):
            rx_buffer_size(0, 0)

    def test_wide_non_posted_credits_fit(self):
        # ndw = 8, default Non-Posted credits (8 header, 8 data): 64 DWs of content, but a TLP starts
        # on a new beat and its last beat can be partial. With b's application stalled, a sends
        # everything the credits allow; all of it must fit in b's Non-Posted buffer (a Nak here would
        # replay into a buffer that is still full), then be delivered in order once b is released.
        ndw    = 8
        bursts = [
            # 1 two-beat TLP, 7 one-beat TLPs: 8 header and 8 data credits, 9 beats.
            [np_wide_tlp(0)] + [np_cfgwr_tlp(i) for i in range(1, 7)] + [np_mrd64_tlp(7)],
            # Worst case for the credits: 4 two-beat TLPs (8 data credits), 4 one-beat TLPs: 12 beats.
            [np_wide_tlp(0x10 + i) for i in range(4)] + [np_mrd64_tlp(0x14 + i) for i in range(4)],
        ]
        dut   = WideDLLLink(ndw)
        outs  = []
        stats = dict(buffered=[])
        done  = []
        def stimulus():
            for _ in range(4):
                yield
            yield dut.link_up.eq(1)
            for _ in range(6000):
                if (yield dut.a.dl_active) and (yield dut.b.dl_active):
                    break
                yield
            else:
                return "DL_Active not reached"
            sent = 0
            for burst in bursts:
                # b's application stalled: the TLPs pile up in b's Non-Posted buffer.
                yield dut.b.source.ready.eq(0)
                for words in burst:
                    beats = [words[i:i+ndw] for i in range(0, len(words), ndw)]
                    for i, beat in enumerate(beats):
                        yield dut.a.sink.data.eq(sum(dw << (32*j) for j, dw in enumerate(beat)))
                        yield dut.a.sink.dws.eq(len(beat))
                        yield dut.a.sink.first.eq(i == 0)
                        yield dut.a.sink.last.eq(i == len(beats) - 1)
                        yield dut.a.sink.valid.eq(1)
                        yield
                        for _ in range(3000):
                            if (yield dut.a.sink.ready):
                                break
                            yield
                        else:
                            return "a.sink not ready (transmitter stuck)"
                    yield dut.a.sink.valid.eq(0)
                sent += len(burst)
                # All of them accepted by b's receiver (and acknowledged), b still stalled.
                buffered = False
                for _ in range(3000):
                    if (yield dut.b.rx.ack_seq) == sent - 1 and not (yield dut.a.tx.replay.pending):
                        buffered = True
                        break
                    yield
                stats["buffered"].append(buffered)
                # Released: delivered, the credits come back for the next burst.
                yield dut.b.source.ready.eq(1)
                for _ in range(3000):
                    if len(outs) >= sent:
                        break
                    yield
            for _ in range(64):
                yield
            return "ok"
        def generator():
            status = yield from stimulus()
            stats["rx_errors"] = (yield dut.b.rx.errors)
            stats["replays"]   = (yield dut.a.tx.replay.replays)
            done.append(status)
        def monitor():
            cur = []
            yield dut.a.source.ready.eq(1)
            while not done:
                if (yield dut.b.source.valid) and (yield dut.b.source.ready):
                    data = (yield dut.b.source.data)
                    if (yield dut.b.source.first):
                        cur = []
                    cur += [(data >> (32*j)) & 0xffffffff for j in range((yield dut.b.source.dws))]
                    if (yield dut.b.source.last):
                        outs.append(list(cur))
                yield
        run_simulation(dut, [generator(), monitor()])
        self.assertEqual(stats["rx_errors"], 0, "TLPs within the advertised credits were Nak'd")
        self.assertEqual(stats["replays"], 0)
        self.assertEqual(done, ["ok"])
        self.assertEqual(stats["buffered"], [True]*len(bursts), "TLPs within the credits not all buffered")
        self.assertEqual(outs, [words for burst in bursts for words in burst])

    def run_link(self, tlps, corrupt=(), drop_acks=None, drop_cycles=400, budget=6000):
        """Send tlps from a to b; corrupt: indexes of the TLPs corrupted on the lane; drop_acks:
        (start, end) TLP indexes during which the DLLPs from b are dropped, then drop_cycles more. Returns the TLPs received
        by b and the link statistics."""
        dut   = DLLLink()
        outs  = []
        stats = {}
        done  = []
        def generator():
            for _ in range(4):
                yield
            yield dut.link_up.eq(1)
            for cycle in range(budget):
                if (yield dut.a.dl_active) and (yield dut.b.dl_active):
                    stats["dl_active"] = cycle
                    break
                yield
            else:
                stats["dl_active"] = None
                done.append(True)
                return
            for i, words in enumerate(tlps):
                if drop_acks is not None:
                    yield dut.drop_dllp_b2a.eq(int(drop_acks[0] <= i < drop_acks[1]))
                if i in corrupt:
                    yield dut.corrupt_a2b.eq(1)
                    yield
                    yield dut.corrupt_a2b.eq(0)
                for j, word in enumerate(words):
                    yield dut.a.sink.data.eq(word)
                    yield dut.a.sink.first.eq(j == 0)
                    yield dut.a.sink.last.eq(j == len(words) - 1)
                    yield dut.a.sink.valid.eq(1)
                    yield
                    while not (yield dut.a.sink.ready):
                        yield
                yield dut.a.sink.valid.eq(0)
            yield dut.a.sink.valid.eq(0)
            if drop_acks is not None:
                # All the Acks lost long enough for the REPLAY_TIMER to expire.
                for _ in range(drop_cycles):
                    yield
            yield dut.drop_dllp_b2a.eq(0)
            for cycle in range(budget):
                if len(outs) >= len(tlps) and not (yield dut.a.tx.replay.pending):
                    for _ in range(64):
                        yield
                    break
                yield
            stats["replays"]   = (yield dut.a.tx.replay.replays)
            stats["rx_errors"] = (yield dut.b.rx.errors)
            stats["pending"]   = (yield dut.a.tx.replay.pending)
            done.append(True)
        def monitor():
            current = []
            yield dut.b.source.ready.eq(1)
            yield dut.a.source.ready.eq(1)
            while not done:
                if (yield dut.b.source.valid):
                    if (yield dut.b.source.first):
                        current = []
                    current.append((yield dut.b.source.data))
                    if (yield dut.b.source.last):
                        outs.append(list(current))
                yield
        def dllp_monitor():
            # DLLPs from b, as they are on the lane.
            dllps = stats["dllps"] = []
            first = None
            while not done:
                if (yield dut.b.phy_sink.valid):
                    data = (yield dut.b.phy_sink.data)
                    if first is not None:
                        dllps.append(first + (data & 0xffffff).to_bytes(3, "little"))
                        first = None
                    elif ((yield dut.b.phy_sink.ctrl) & 1) and (data & 0xff) == SDP.value:
                        first = (data >> 8).to_bytes(3, "little")
                yield
        run_simulation(dut, [generator(), monitor(), dllp_monitor()])
        return outs, stats

    def test_init_and_tlps(self):
        rng  = random.Random(0)
        tlps = [mwr_tlp(rng, rng.randrange(1, 9), i) for i in range(10)]
        outs, stats = self.run_link(tlps)
        self.assertIsNotNone(stats["dl_active"], "DL_Active not reached")
        self.assertEqual(outs, tlps)
        self.assertEqual(stats["replays"], 0)
        self.assertEqual(stats["pending"], 0)

    def test_corrupted_tlp_replayed(self):
        # Corrupted TLPs: Nak, replayed, received exactly once in order.
        rng  = random.Random(1)
        tlps = [mwr_tlp(rng, rng.randrange(1, 6), i) for i in range(8)]
        outs, stats = self.run_link(tlps, corrupt={2})
        self.assertIsNotNone(stats["dl_active"])
        self.assertEqual(outs, tlps)
        self.assertGreaterEqual(stats["replays"], 1)
        self.assertGreaterEqual(stats["rx_errors"], 1)
        self.assertEqual(stats["pending"], 0)

    def test_lost_acks_replay_timer(self):
        # Acks dropped for a while: REPLAY_TIMER replays, duplicates discarded by the receiver.
        rng  = random.Random(2)
        tlps = [mwr_tlp(rng, rng.randrange(1, 4), i) for i in range(8)]
        outs, stats = self.run_link(tlps, drop_acks=(0, 8), budget=8000)
        self.assertIsNotNone(stats["dl_active"])
        self.assertEqual(outs, tlps)
        self.assertGreaterEqual(stats["replays"], 1)
        self.assertEqual(stats["pending"], 0)

    def test_completion_overflow_dropped_not_naked(self):
        # Completion credit is infinite, so a receiver can be sent more Completions than its buffer
        # holds. With b's application stalled, the Completion buffer fills and the surplus Completions
        # must be Ack'd and dropped (a Nak would replay them forever on a full buffer), the drops
        # counted, and no replay triggered. A Posted TLP sent afterwards must still be delivered.
        rng    = random.Random(5)
        cpls   = [cpl_tlp(i & 0xff) for i in range(60)] # 60 x 3 DW > b's 128-DW Completion buffer.
        posted = mwr_tlp(rng, 2, 0xaa)
        dut    = DLLLink()
        outs   = []
        stats  = {}
        done   = []
        def generator():
            for _ in range(4):
                yield
            yield dut.link_up.eq(1)
            for _ in range(6000):
                if (yield dut.a.dl_active) and (yield dut.b.dl_active):
                    break
                yield
            # b's application stalled: the Completions pile up in b's receive buffer.
            yield dut.b.source.ready.eq(0)
            for words in cpls:
                for j, word in enumerate(words):
                    yield dut.a.sink.data.eq(word)
                    yield dut.a.sink.first.eq(j == 0)
                    yield dut.a.sink.last.eq(j == len(words) - 1)
                    yield dut.a.sink.valid.eq(1)
                    yield
                    while not (yield dut.a.sink.ready):
                        yield
                yield dut.a.sink.valid.eq(0)
            for _ in range(1000):
                yield
            stats["cpl_drops"] = (yield dut.b.rx.cpl_drops)
            stats["rx_errors"] = (yield dut.b.rx.errors)
            # Drain the buffered Completions, then a Posted TLP must still get through.
            yield dut.b.source.ready.eq(1)
            for j, word in enumerate(posted):
                yield dut.a.sink.data.eq(word)
                yield dut.a.sink.first.eq(j == 0)
                yield dut.a.sink.last.eq(j == len(posted) - 1)
                yield dut.a.sink.valid.eq(1)
                yield
                while not (yield dut.a.sink.ready):
                    yield
            yield dut.a.sink.valid.eq(0)
            for _ in range(3000):
                if (posted in outs) and not (yield dut.a.tx.replay.pending):
                    break
                yield
            stats["replays"] = (yield dut.a.tx.replay.replays)
            stats["pending"] = (yield dut.a.tx.replay.pending)
            done.append(True)
        def monitor():
            cur = []
            yield dut.a.source.ready.eq(1)
            while not done:
                if (yield dut.b.source.valid) and (yield dut.b.source.ready):
                    if (yield dut.b.source.first):
                        cur = []
                    cur.append((yield dut.b.source.data))
                    if (yield dut.b.source.last):
                        outs.append(list(cur))
                yield
        run_simulation(dut, [generator(), monitor()])
        self.assertGreater(stats["cpl_drops"], 0, "no Completion was dropped (buffer did not overflow)")
        self.assertEqual(stats["rx_errors"], 0, "a drop must not be a receive error")
        self.assertEqual(stats["replays"], 0, "dropped Completions must be Ack'd, not Nak'd/replayed")
        self.assertEqual(stats["pending"], 0)
        self.assertIn(posted, outs, "the Posted TLP was not delivered after the Completion overflow")

    def test_dllps_under_load(self):
        # Ack sequence number and credits changing all the time: each DLLP is consistent on the lane
        # and one sent while another one is requested doesn't take its place (no Ack lost, no replay).
        rng  = random.Random(3)
        tlps = [mwr_tlp(rng, rng.randrange(1, 4), i) for i in range(64)]
        outs, stats = self.run_link(tlps, budget=20000)
        self.assertIsNotNone(stats["dl_active"])
        self.assertEqual(outs, tlps)
        self.assertGreater(len(stats["dllps"]), 64)
        bad = [dllp.hex() for dllp in stats["dllps"] if dllp_crc_bytes(dllp[:4]) != dllp[4:]]
        self.assertEqual(bad, [])
        self.assertEqual(stats["replays"], 0)
        self.assertEqual(stats["pending"], 0)
