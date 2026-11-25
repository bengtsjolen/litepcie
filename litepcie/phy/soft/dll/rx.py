#
# This file is part of LitePCIe.
#
# Copyright (c) 2026 Florent Kermarrec <florent@enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Soft PCIe PHY Data Link Layer receiver (PCIe Base 3.5.3): TLPs from the lane words (RXFramer),
LCRC and sequence number checks, store-and-forward to the Transaction Layer, Ack/Nak scheduling,
received DLLPs, credits of the delivered TLPs."""

from migen import *

from litex.gen import *

from litex.soc.interconnect import stream

from litepcie.phy.soft.common           import lane_layout
from litepcie.phy.soft.dll.crc          import LCRC
from litepcie.phy.soft.dll.framing      import RXFramer, seq_symbols
from litepcie.phy.soft.dll.buffer       import PacketBuffer
from litepcie.phy.soft.dll.flow_control import tlp_credits, FC_P, FC_NP, FC_CPL, RX_TAG_WIDTH

# DLL RX -------------------------------------------------------------------------------------------

class DLLRX(LiteXModule):
    # depth/packets size the Completion buffer (the bulk of the receive memory); the Posted and
    # Non-Posted buffers are sized to their advertised credits (see DataLinkLayer).
    def __init__(self, depth=2048, packets=32,
                 p_depth=1024, p_packets=16, np_depth=128, np_packets=16):
        self.reset        = Signal()                          # i
        self.sink         = sink   = stream.Endpoint(lane_layout(4))   # i: lane words (L0).
        # One receive buffer per flow-control type, so a Completion the Transaction Layer does not
        # accept cannot block a (Config) Non-Posted request behind it. Each source carries the accept
        # tag (wire order), so RXAcceptMerge can reorder them into one stream. Assigned to the
        # PacketBuffer sources below.
        self.source_p     = None       # o: Posted TLPs.
        self.source_np    = None       # o: Non-Posted TLPs (Config, memory reads, UR requests).
        self.source_cpl   = None       # o: Completions.
        self.dllp         = Signal(48) # o
        self.dllp_valid   = Signal()   # o (Pulse)
        self.ack_req      = Signal()   # o (Pulse): schedule an Ack (ack_seq).
        self.nak_req      = Signal()   # o (Pulse): schedule a Nak (ack_seq).
        self.ack_seq      = Signal(12) # o: NEXT_RCV_SEQ - 1.
        # Per-type delivery pulses for the credit accounting (Completions are infinite: not tracked).
        self.deliver_p       = Signal()   # o (Pulse): Posted TLP delivered.
        self.deliver_np      = Signal()   # o (Pulse): Non-Posted TLP delivered.
        self.deliver_p_data  = Signal(9)  # o
        self.deliver_np_data = Signal(9)  # o
        self.errors       = Signal(16) # o: LCRC/sequence/framing errors.
        self.cpl_drops    = Signal(16) # o: Completions Ack'd but dropped (buffer full, see below).

        # # #

        # Framing.
        self.framer = framer = RXFramer()
        self.comb += [
            sink.connect(framer.sink),
            self.dllp.eq(framer.dllp),
            self.dllp_valid.eq(framer.dllp_valid),
        ]

        # LCRC over the Sequence Number and the TLP DWs (the last DW is the received LCRC).
        self.lcrc = lcrc = LCRC(data_width=64, sizes=[4, 6])
        tlp = framer.tlp
        self.comb += [
            tlp.ready.eq(1),
            lcrc.ce.eq(tlp.valid & ~tlp.last),
            If(tlp.first,
                lcrc.data.eq(Cat(seq_symbols(tlp.seq), tlp.data)),
                lcrc.bytes.eq(6),
            ).Else(
                lcrc.data.eq(tlp.data),
                lcrc.bytes.eq(4),
            ),
            lcrc.clear.eq(self.reset | (tlp.valid & tlp.last)),
        ]

        # Store-and-forward buffers (one per flow-control type), the TLP DWs delayed by one word (the
        # LCRC DW is not stored). The TLP's type (sampled on its first word) selects its buffer. Each
        # buffer also stores the accept tag (wire order) of its committed TLPs (see below).
        param_layout = [("fc_data", 9), ("tag", RX_TAG_WIDTH)]
        self.buf_p   = buf_p   = PacketBuffer(param_layout, depth=p_depth,  packets=p_packets)
        self.buf_np  = buf_np  = PacketBuffer(param_layout, depth=np_depth, packets=np_packets)
        self.buf_cpl = buf_cpl = PacketBuffer(param_layout, depth=depth,    packets=packets)
        buffers = [buf_p, buf_np, buf_cpl]
        statements, dw0_type, dw0_data = tlp_credits(tlp.data)
        self.comb += statements
        fc_type  = Signal(2)
        fc_data  = Signal(9)
        prev_valid = Signal()
        prev_data  = Signal(32)
        prev_first = Signal()
        prev_seq   = Signal(12)
        lost_pnp   = Signal() # Posted/Non-Posted word not accepted by its buffer (will Nak).
        lost_cpl   = Signal() # Completion word not accepted (infinite credit: drop, do not Nak).
        lost_pnp_r = Signal() # Posted/Non-Posted words lost in the current TLP: Nak at its end.
        lost_cpl_r = Signal() # Completion words lost in the current TLP: drop at its end.
        end_r      = Signal()
        self.sync += [
            If(self.reset | end_r | framer.tlp_abort,
                lost_pnp_r.eq(0)
            ).Elif(lost_pnp,
                lost_pnp_r.eq(1)
            ),
            If(self.reset | end_r | framer.tlp_abort,
                lost_cpl_r.eq(0)
            ).Elif(lost_cpl,
                lost_cpl_r.eq(1)
            ),
            If(self.reset | (tlp.valid & tlp.last) | framer.tlp_abort,
                prev_valid.eq(0)
            ).Elif(tlp.valid,
                prev_valid.eq(1),
                prev_data.eq(tlp.data),
                prev_first.eq(tlp.first),
                If(tlp.first,
                    prev_seq.eq(tlp.seq),
                    fc_type.eq(dw0_type),
                    fc_data.eq(dw0_data),
                ),
            )
        ]
        # Checks at the LCRC DW, registered (timing): the last DW is written and the TLP committed or
        # dropped one cycle after the LCRC DW. The first DW of the next TLP comes at least 2 lane words
        # later (END, STP, Sequence Number), so prev_data still holds the last DW and nothing collides.
        next_seq      = Signal(12)
        nak_scheduled = Signal()
        seq_diff      = Signal(12)
        end           = tlp.valid & tlp.last
        prev_valid_r  = Signal()
        lcrc_ok_r     = Signal()
        edb_r         = Signal()
        in_order_r    = Signal()
        duplicate_r   = Signal()
        self.comb += seq_diff.eq(next_seq - prev_seq)
        self.sync += [
            end_r.eq(end & ~self.reset),
            prev_valid_r.eq(prev_valid),
            lcrc_ok_r.eq(Mux(tlp.edb, lcrc.value == ~tlp.data, lcrc.value == tlp.data)),
            edb_r.eq(tlp.edb),
            in_order_r.eq(seq_diff == 0),
            duplicate_r.eq((seq_diff != 0) & (seq_diff < 2048)),
        ]
        accept = Signal()
        bad    = Signal()
        # One word written, steered to the current TLP's buffer by its sampled type.
        word_valid = Signal()
        sel_ready  = Signal()
        self.comb += [
            word_valid.eq((tlp.valid & prev_valid & ~tlp.last) | (end_r & prev_valid_r)),
            Case(fc_type, {t: sel_ready.eq(buffers[t].sink.ready) for t in range(3)}),
        ]
        # Accept tag (wire order): every TLP committed to a buffer is stamped with a monotonic index,
        # incremented only on a commit. A dropped Completion is not committed and takes no tag (using
        # the link Sequence Number would leave a hole and stall the merge).
        tag = Signal(RX_TAG_WIDTH)
        for t, buf in enumerate(buffers):
            self.comb += [
                buf.sink.valid.eq(word_valid & (fc_type == t)),
                buf.sink.data.eq(prev_data),
                buf.sink.first.eq(prev_first),
                buf.sink.last.eq(end_r),
                buf.sink.fc_data.eq(fc_data),
                buf.sink.tag.eq(tag),
            ]
        committed = Signal()
        self.comb += committed.eq(Reduce("OR",
            [buf.sink.valid & buf.sink.ready & buf.sink.last & ~buf.abort for buf in buffers]))
        self.comb += [
            lost_pnp.eq(word_valid & (fc_type != FC_CPL) & ~sel_ready),
            lost_cpl.eq(word_valid & (fc_type == FC_CPL) & ~sel_ready),
            # A lost Completion is dropped, not Nak'd: Completion credit is infinite, so a Nak would
            # replay it forever. A lost Posted/Non-Posted word is Nak'd (and should not happen: those
            # types are gated by the credits we advertise, matching their buffer depth).
            accept.eq(end_r & prev_valid_r & lcrc_ok_r & ~edb_r & in_order_r & ~lost_pnp & ~lost_pnp_r),
            bad.eq(end_r & (~prev_valid_r | ~lcrc_ok_r | lost_pnp | lost_pnp_r | (~edb_r & ~in_order_r & ~duplicate_r))),
            buf_p.abort.eq(  (fc_type == FC_P)   & (framer.tlp_abort | (end_r & ~accept))),
            buf_np.abort.eq( (fc_type == FC_NP)  & (framer.tlp_abort | (end_r & ~accept))),
            buf_cpl.abort.eq((fc_type == FC_CPL) & (framer.tlp_abort | (end_r & (~accept | lost_cpl | lost_cpl_r)))),
            self.ack_seq.eq(next_seq - 1),
            self.ack_req.eq(accept | (end_r & lcrc_ok_r & ~edb_r & duplicate_r)),
            self.nak_req.eq((bad | framer.tlp_abort | framer.error) & ~nak_scheduled),
        ]
        # A Completion that passed its checks (Ack'd) but did not fit in its buffer is dropped (its
        # credit is infinite, a Nak would replay it forever): silent DMA data loss, so it is counted.
        cpl_drop = Signal()
        self.comb += cpl_drop.eq(accept & (fc_type == FC_CPL) & (lost_cpl | lost_cpl_r))
        self.sync += [
            If(self.reset,
                next_seq.eq(0),
                nak_scheduled.eq(0),
                self.errors.eq(0),
                self.cpl_drops.eq(0),
            ).Else(
                If(accept,
                    next_seq.eq(next_seq + 1),
                    nak_scheduled.eq(0),
                ),
                If(self.nak_req,
                    nak_scheduled.eq(1),
                    self.errors.eq(self.errors + 1),
                ),
                If(cpl_drop, self.cpl_drops.eq(self.cpl_drops + 1)),
            )
        ]
        # The accept tag is free-running (not reset with the link): it stays in step with the merge's
        # next-tag counter, which also only advances on a delivery.
        self.sync += If(committed, tag.eq(tag + 1))

        # Transaction Layer delivery and credits.
        # Each buffer's source is exposed to the Transaction Layer independently. A delivery (last
        # word consumed) returns that type's credits; Completions are infinite and not tracked.
        self.source_p   = buf_p.source
        self.source_np  = buf_np.source
        self.source_cpl = buf_cpl.source
        self.comb += [
            self.deliver_p.eq(buf_p.source.valid & buf_p.source.ready & buf_p.source.last),
            self.deliver_np.eq(buf_np.source.valid & buf_np.source.ready & buf_np.source.last),
            self.deliver_p_data.eq(buf_p.source.fc_data),
            self.deliver_np_data.eq(buf_np.source.fc_data),
        ]
