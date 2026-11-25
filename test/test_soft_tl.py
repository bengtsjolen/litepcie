#
# This file is part of LitePCIe.
#
# Copyright (c) 2026 Florent Kermarrec <florent@enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from migen import *

from litepcie.phy.soft.tl.adapter      import TXAdapter, RXAdapter
from litepcie.phy.soft.tl.config_space import PCIeConfigSpace
from litepcie.phy.soft.tl.core         import SoftTransactionLayer
from litepcie.phy.soft.tl.wide         import WideTransactionLayer

MSI_CTRL   = 0x48
MSI_ADR_LO = 0x4c
MSI_DATA   = 0x54

def swap(dw):
    return int.from_bytes(dw.to_bytes(4, "little"), "big")

class TestSoftTL(unittest.TestCase):
    def test_tx_adapter(self):
        # Beats to DWs: be=0 DWs dropped, last on the last valid DW, byte swap.
        for data_width, beats in [
            (64,  [(0x0000000200000001, 0xff, 0), (0x0000000000000003, 0x0f, 1)]),
            (64,  [(0x0000000200000001, 0xff, 0), (0x0000000400000003, 0xff, 1)]),
            (128, [(0x00000000000000030000000200000001, 0x0fff, 1)]),
            (128, [(0x00000004000000030000000200000001, 0xffff, 0), (0x5, 0x000f, 1)]),
        ]:
            dut = TXAdapter(data_width)
            out = []
            def generator():
                yield dut.source.ready.eq(1)
                for dat, be, last in beats:
                    yield dut.sink.dat.eq(dat)
                    yield dut.sink.be.eq(be)
                    yield dut.sink.last.eq(last)
                    yield dut.sink.valid.eq(1)
                    yield
                    while True:
                        if (yield dut.source.valid):
                            out.append(((yield dut.source.data), (yield dut.source.first), (yield dut.source.last)))
                        if (yield dut.sink.ready):
                            break
                        yield
                yield dut.sink.valid.eq(0)
                yield
            run_simulation(dut, generator())
            n = sum(bin(be).count("1")//4 for dat, be, last in beats)
            self.assertEqual([d for d, f, l in out], [swap(i + 1) for i in range(n)])
            self.assertEqual([f for d, f, l in out], [1] + [0]*(n - 1))
            self.assertEqual([l for d, f, l in out], [0]*(n - 1) + [1])

    def test_rx_adapter(self):
        # DWs to beats: be=0 past the end, a TLP starts on DW 0.
        for data_width, lengths in [(64, [3, 4, 5]), (128, [3, 4, 5, 8])]:
            dut = RXAdapter(data_width)
            ratio = data_width//32
            beats = []
            def generator():
                yield dut.source.ready.eq(1)
                for length in lengths:
                    for i in range(length):
                        yield dut.sink.data.eq(swap(i + 1))
                        yield dut.sink.first.eq(i == 0)
                        yield dut.sink.last.eq(i == length - 1)
                        yield dut.sink.valid.eq(1)
                        yield
                        if (yield dut.source.valid):
                            beats.append(((yield dut.source.dat), (yield dut.source.be), (yield dut.source.last)))
                yield dut.sink.valid.eq(0)
                yield
                if (yield dut.source.valid):
                    beats.append(((yield dut.source.dat), (yield dut.source.be), (yield dut.source.last)))
                yield
            run_simulation(dut, generator())
            expected = []
            for length in lengths:
                for b in range(0, length, ratio):
                    count = min(ratio, length - b)
                    dat = sum((b + j + 1) << 32*j for j in range(count))
                    be  = (1 << 4*count) - 1
                    expected.append((dat, be, int(b + ratio >= length)))
            self.assertEqual(beats, expected)

    def test_config_space(self):
        dut = PCIeConfigSpace(bar0_mask=0xfff00000)
        def read(adr):
            yield dut.adr.eq(adr//4)
            yield
            return (yield dut.dat_r)
        def write(adr, value, be=0xf):
            yield dut.adr.eq(adr//4)
            yield dut.dat_w.eq(value)
            yield dut.be.eq(be)
            yield dut.we.eq(1)
            yield
            yield dut.we.eq(0)
            yield
        def generator():
            self.assertEqual((yield from read(0x00)), 0x702110ee)
            self.assertEqual((yield from read(0x100)), 0) # No extended capabilities.
            # Capability list ends.
            ptr = (yield from read(0x34)) & 0xff
            caps = []
            while ptr:
                dw = yield from read(ptr)
                caps.append(dw & 0xff)
                ptr = (dw >> 8) & 0xff
            self.assertEqual(caps, [0x01, 0x05, 0x10])
            # BAR sizing.
            yield from write(0x10, 0xffffffff)
            self.assertEqual((yield from read(0x10)), 0xfff00000)
            # Command with byte enables, read-only bits kept.
            yield from write(0x04, 0xffffffff, be=0x1)
            self.assertEqual((yield from read(0x04)), 0x00100046)
            self.assertEqual((yield dut.mse), 1)
            self.assertEqual((yield dut.bme), 1)
            # Device Control: MPS/MRRS reset and write.
            self.assertEqual((yield dut.mps), 0)
            self.assertEqual((yield dut.mrrs), 2)
            yield from write(0x68, 0x00001030)
            self.assertEqual((yield dut.mps), 1)
            self.assertEqual((yield dut.mrrs), 1)
            # No power feature to enable: Aux Power PM (Device Control), ASPM Control and Clock PM (Link
            # Control) hardwired to 0, no ASPM or L0s/L1 latencies in Link Capabilities, no PME in PMC.
            yield from write(0x68, 0x00000700, be=0x2)
            self.assertEqual((yield from read(0x68)) & 0x0700, 0)
            yield from write(0x70, 0x00000103, be=0x3)
            self.assertEqual((yield from read(0x70)) & 0x0103, 0)
            self.assertEqual((yield from read(0x6c)) & 0x0003fc00, 0)
            self.assertEqual((yield from read(0x40)) >> 27, 0)
            yield from write(0x68, 0x00001030)
            # Unsupported Request Detected: set, write 1 to clear.
            yield dut.ur_detected.eq(1)
            yield
            yield dut.ur_detected.eq(0)
            yield
            self.assertEqual(((yield from read(0x68)) >> 19) & 1, 1)
            yield from write(0x68, 0x00080000, be=0x4)
            self.assertEqual(((yield from read(0x68)) >> 19) & 1, 0)
            # MSI.
            yield from write(0x4c, 0xfee00003)
            yield from write(0x50, 0x00000001)
            yield from write(0x54, 0xffff1234)
            yield from write(0x48, 0x00010000)
            self.assertEqual((yield dut.msi_enable), 1)
            self.assertEqual((yield dut.msi_address), 0x1fee00000)
            self.assertEqual((yield dut.msi_data), 0x1234)
        run_simulation(dut, generator())

    def test_config_space_gen3(self):
        # 8.0 GT/s: speeds in Link Capabilities (2), Secondary PCI Express Extended Capability at 0x100.
        dut = PCIeConfigSpace(bar0_mask=0xfff00000, nlanes=8, gen2_capable=True, gen3_capable=True)
        def read(adr):
            yield dut.adr.eq(adr//4)
            yield
            return (yield dut.dat_r)
        def generator():
            self.assertEqual((yield from read(0x6c)) & 0x3ff, 0x083) # x8, 8.0 GT/s.
            self.assertEqual((yield from read(0x8c)), 0b1110)
            self.assertEqual((yield from read(0x90)) & 0xf, 3)
            self.assertEqual((yield from read(0x100)), 0x00010019)
            lanes_eq = []
            for i in range(5):
                lanes_eq.append((yield from read(0x10c + 4*i)))
            self.assertEqual(lanes_eq, [0x7f7f7f7f]*4 + [0])
        run_simulation(dut, generator())

    def test_config_not_blocked_by_stalled_completion(self):
        # The whole point of the per-type receive queues: a Completion the application will not accept
        # (ep_source stalled) must not block a Config read. The config completion still goes out on the
        # transmit path, and the held Completion is delivered unchanged once the application is ready.
        dut    = SoftTransactionLayer(bar0_mask=0xfff00000)
        cpl_in = [0x0000000a, 0x11111111, 0x22222222]  # A 3-DW Completion waiting on sink_cpl.
        cfg_rd = [0x01000004, 0x0f000000, 0x00000000]  # CfgRd0 to function 0 (3 DW, no data).
        tx, ep = [], []
        @passive
        def drive_cpl():
            i = 0
            yield dut.sink_cpl.valid.eq(1)
            while i < len(cpl_in):
                yield dut.sink_cpl.data.eq(cpl_in[i])
                yield dut.sink_cpl.first.eq(i == 0)
                yield dut.sink_cpl.last.eq(i == len(cpl_in) - 1)
                yield
                if (yield dut.sink_cpl.valid) and (yield dut.sink_cpl.ready):
                    i += 1
            yield dut.sink_cpl.valid.eq(0)
        @passive
        def collect_tx():
            yield dut.source.ready.eq(1)
            while True:
                if (yield dut.source.valid) and (yield dut.source.ready):
                    tx.append((yield dut.source.data))
                yield
        @passive
        def collect_ep():
            while True:
                if (yield dut.ep_source.valid) and (yield dut.ep_source.ready):
                    ep.append((yield dut.ep_source.data))
                yield
        def generator():
            yield dut.ep_source.ready.eq(0)  # Application cannot accept the Completion.
            for _ in range(8):
                yield
            yield dut.sink_np.valid.eq(1)
            for i, dw in enumerate(cfg_rd):
                yield dut.sink_np.data.eq(dw)
                yield dut.sink_np.first.eq(i == 0)
                yield dut.sink_np.last.eq(i == len(cfg_rd) - 1)
                yield
                while not (yield dut.sink_np.ready):
                    yield
            yield dut.sink_np.valid.eq(0)
            yield dut.sink_np.first.eq(0)
            yield dut.sink_np.last.eq(0)
            for _ in range(300):
                if len(tx) >= 4:
                    break
                yield
            self.assertGreaterEqual(len(tx), 4, "config completion did not appear while app stalled")
            self.assertEqual(len(ep), 0, "the Completion was delivered though the app was stalled")
            yield dut.ep_source.ready.eq(1)  # Release the application.
            for _ in range(300):
                if len(ep) >= len(cpl_in):
                    break
                yield
            self.assertEqual(ep[:len(cpl_in)], cpl_in, "the held Completion was not delivered unchanged")
        run_simulation(dut, [generator(), drive_cpl(), collect_tx(), collect_ep()])

    def test_config_not_blocked_by_stalled_completion_wide(self):
        # As above, on the wide Transaction Layer (ndw=2): beats carry dws.
        ndw    = 2
        dut    = WideTransactionLayer(ndw, bar0_mask=0xfff00000)
        cpl_in = [0x0000000a, 0x11111111, 0x22222222]
        cfg_rd = [0x01000004, 0x0f000000, 0x00000000]
        def to_beats(dws):
            beats = []
            for i in range(0, len(dws), ndw):
                chunk = dws[i:i + ndw]
                beats.append((sum(d << (32*j) for j, d in enumerate(chunk)), len(chunk), i + ndw >= len(dws)))
            return beats
        tx, ep = [], []
        def drain(data, n):
            return [(data >> (32*j)) & 0xffffffff for j in range(n)]
        @passive
        def drive_cpl():
            beats = to_beats(cpl_in)
            i = 0
            yield dut.sink_cpl.valid.eq(1)
            while i < len(beats):
                data, n, last = beats[i]
                yield dut.sink_cpl.data.eq(data)
                yield dut.sink_cpl.dws.eq(n)
                yield dut.sink_cpl.first.eq(i == 0)
                yield dut.sink_cpl.last.eq(last)
                yield
                if (yield dut.sink_cpl.valid) and (yield dut.sink_cpl.ready):
                    i += 1
            yield dut.sink_cpl.valid.eq(0)
        @passive
        def collect_tx():
            yield dut.source.ready.eq(1)
            while True:
                if (yield dut.source.valid) and (yield dut.source.ready):
                    tx.extend(drain((yield dut.source.data), (yield dut.source.dws)))
                yield
        @passive
        def collect_ep():
            while True:
                if (yield dut.ep_source.valid) and (yield dut.ep_source.ready):
                    ep.extend(drain((yield dut.ep_source.data), (yield dut.ep_source.dws)))
                yield
        def generator():
            yield dut.ep_source.ready.eq(0)
            for _ in range(8):
                yield
            for i, (data, n, last) in enumerate(to_beats(cfg_rd)):
                yield dut.sink_np.valid.eq(1)
                yield dut.sink_np.data.eq(data)
                yield dut.sink_np.dws.eq(n)
                yield dut.sink_np.first.eq(i == 0)
                yield dut.sink_np.last.eq(last)
                yield
                while not (yield dut.sink_np.ready):
                    yield
            yield dut.sink_np.valid.eq(0)
            yield dut.sink_np.first.eq(0)
            yield dut.sink_np.last.eq(0)
            for _ in range(400):
                if len(tx) >= 4:
                    break
                yield
            self.assertGreaterEqual(len(tx), 4, "config completion did not appear while app stalled")
            self.assertEqual(len(ep), 0, "the Completion was delivered though the app was stalled")
            yield dut.ep_source.ready.eq(1)
            for _ in range(400):
                if len(ep) >= len(cpl_in):
                    break
                yield
            self.assertEqual(ep[:len(cpl_in)], cpl_in, "the held Completion was not delivered unchanged")
        run_simulation(dut, [generator(), drive_cpl(), collect_tx(), collect_ep()])

    def test_message_absorbed_mwr_forwarded(self):
        # A Posted Message (byte 0 = 0x34) must be absorbed (no completion, nothing to the application),
        # while a 32-bit Memory Write on the same Posted sink is still forwarded.
        dut = SoftTransactionLayer(bar0_mask=0xfff00000)
        msg = [0x00000034, 0x11111111, 0x22222222, 0x33333333] # Message, byte 0 = 0x34, 4 DW.
        mwr = [0x01000040, 0x0f000001, 0x00001000, 0xdeadbeef] # MWr32 (Fmt 010 Type 0), 1 DW data.
        ep, tx = [], []
        @passive
        def collect_ep():
            yield dut.ep_source.ready.eq(1)
            while True:
                if (yield dut.ep_source.valid) and (yield dut.ep_source.ready):
                    ep.append((yield dut.ep_source.data))
                yield
        @passive
        def collect_tx():
            yield dut.source.ready.eq(1)
            while True:
                if (yield dut.source.valid) and (yield dut.source.ready):
                    tx.append((yield dut.source.data))
                yield
        def drive(words):
            yield dut.sink_p.valid.eq(1)
            for i, dw in enumerate(words):
                yield dut.sink_p.data.eq(dw)
                yield dut.sink_p.first.eq(i == 0)
                yield dut.sink_p.last.eq(i == len(words) - 1)
                yield
                while not (yield dut.sink_p.ready):
                    yield
            yield dut.sink_p.valid.eq(0)
            yield dut.sink_p.first.eq(0)
            yield dut.sink_p.last.eq(0)
        def generator():
            yield from drive(msg)
            for _ in range(32):
                yield
            self.assertEqual(ep, [], "the Message was forwarded to the application")
            yield from drive(mwr)
            for _ in range(64):
                if len(ep) >= len(mwr):
                    break
                yield
            self.assertEqual(ep, mwr, "the 32-bit Memory Write was not forwarded unchanged")
            self.assertEqual(tx, [], "a Posted TLP must not produce a completion")
        run_simulation(dut, [generator(), collect_ep(), collect_tx()])

    def test_msi_not_starved(self):
        # LitePCIe transmits back to back under DMA: an MSI waits for one TLP at most, and it can't be
        # left to the gaps since DMA keeps running without the driver when it misses its interrupts.
        dut = SoftTransactionLayer(bar0_mask=0xfff00000)
        tx  = []
        def cfg_write(reg, value):
            # CfgWr0 is a Non-Posted request: driven on sink_np. 1 DW, First DW BE=0xf.
            yield dut.sink_np.valid.eq(1)
            for i, dw in enumerate([0x01000044, 0x0f000000, reg << 26, value]):
                yield dut.sink_np.data.eq(dw)
                yield dut.sink_np.last.eq(i == 3)
                yield
                while not (yield dut.sink_np.ready):
                    yield
            yield dut.sink_np.valid.eq(0)
            yield dut.sink_np.last.eq(0)
        @passive
        def collect():
            yield dut.source.ready.eq(1)
            while True:
                if (yield dut.source.valid):
                    tx.append(((yield dut.source.data), (yield dut.source.first), (yield dut.source.last)))
                yield
        @passive
        def ep():
            # 4 DW TLPs without gaps.
            yield dut.ep_sink.valid.eq(1)
            index = 0
            while True:
                yield dut.ep_sink.data.eq(0x40000000 | index)
                yield dut.ep_sink.first.eq(index == 0)
                yield dut.ep_sink.last.eq(index == 3)
                yield
                if (yield dut.ep_sink.ready):
                    index = (index + 1)%4
        def generator():
            yield from cfg_write(0x04 >> 2, 0x00000004) # Bus Master Enable.
            yield from cfg_write(MSI_ADR_LO >> 2, 0xfee00000)
            yield from cfg_write(MSI_DATA   >> 2, 0x00001234)
            yield from cfg_write(MSI_CTRL   >> 2, 0x00010000)
            for i in range(32):
                yield
            # Several interrupts, each one has to go through.
            for i in range(4):
                yield dut.msi.valid.eq(1)
                for wait in range(16):
                    yield
                    if (yield dut.msi.ready):
                        break
                else:
                    self.fail("MSI starved by the TLPs from LitePCIe")
                yield dut.msi.valid.eq(0)
                for j in range(64):
                    yield
            # LitePCIe isn't starved in return, and TLPs aren't mixed.
            tlps, tlp = [], []
            for data, first, last in tx:
                tlp = [data] if first else tlp + [data]
                if last:
                    tlps.append(tlp)
            msis = [t for t in tlps if t[-1] == 0x00001234]
            eps  = [t for t in tlps if t[0] >> 16 == 0x4000]
            self.assertEqual(len(msis), 4)
            self.assertTrue(all(len(t) == 4 for t in msis + eps))
            self.assertGreater(len(eps), 50)
        run_simulation(dut, [generator(), collect(), ep()])
