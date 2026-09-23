// Simple IoU tracker (spec §7.3).

import type { Detection } from '../vision/types.js';
import { rectIoU } from '../vision/yunet.js';

export interface Track {
  id: string;
  det: Detection; // last detection
  firstSeen: number; // ms, monotonic
  firstSeenWall: number; // ms, epoch
  lastSeen: number;
  hits: number;
}

export class IouTracker {
  private tracks = new Map<string, Track>();
  private seq = 0;

  constructor(
    private readonly iouThreshold: number,
    private readonly ttlMs: number,
    private readonly idPrefix = 't-',
  ) {}

  /**
   * Associate detections with live tracks (greedy by IoU), create tracks for the rest,
   * expire tracks unseen for longer than ttl. Returns tracks updated in this frame and expired ones.
   */
  update(dets: Detection[], ts: number, wall = Date.now()): { updated: Track[]; expired: Track[] } {
    const expired: Track[] = [];
    for (const [id, t] of this.tracks) {
      if (ts - t.lastSeen > this.ttlMs) {
        this.tracks.delete(id);
        expired.push(t);
      }
    }
    const pairs: { t: Track; d: number; iou: number }[] = [];
    for (const t of this.tracks.values()) {
      dets.forEach((d, i) => {
        const iou = rectIoU(t.det.box, d.box);
        if (iou >= this.iouThreshold) pairs.push({ t, d: i, iou });
      });
    }
    pairs.sort((a, b) => b.iou - a.iou);
    const usedT = new Set<string>();
    const usedD = new Set<number>();
    const updated: Track[] = [];
    for (const p of pairs) {
      if (usedT.has(p.t.id) || usedD.has(p.d)) continue;
      usedT.add(p.t.id);
      usedD.add(p.d);
      p.t.det = dets[p.d];
      p.t.lastSeen = ts;
      p.t.hits++;
      updated.push(p.t);
    }
    dets.forEach((d, i) => {
      if (usedD.has(i)) return;
      const t: Track = {
        id: `${this.idPrefix}${String(++this.seq).padStart(6, '0')}`,
        det: d,
        firstSeen: ts,
        firstSeenWall: wall,
        lastSeen: ts,
        hits: 1,
      };
      this.tracks.set(t.id, t);
      updated.push(t);
    });
    return { updated, expired };
  }

  live(): Track[] {
    return [...this.tracks.values()];
  }

  clear(): void {
    this.tracks.clear();
  }
}
