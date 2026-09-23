// In-memory gallery (spec §8.3, §8.6): one contiguous Float32Array N×D, brute-force cosine.

export interface MatchCandidate {
  personId: string;
  photoId: string; // best-scoring photo of that person
  score: number;
}

export interface GalleryRow {
  personId: string;
  photoId: string;
}

export interface Gallery {
  match(probe: Float32Array, topK: number): MatchCandidate[];
  upsertPhoto(personId: string, photoId: string, emb: Float32Array): void;
  removePhoto(personId: string, photoId: string): void;
  removePerson(personId: string): void;
  setPersonActive(personId: string, active: boolean): void;
  stats(): { persons: number; embeddings: number };
}

export class GalleryIndex implements Gallery {
  private data: Float32Array;
  private capacity: number;
  private n = 0;
  private rows: GalleryRow[] = [];
  private rowOrd: Int32Array; // row -> person ordinal
  private readonly rowOf = new Map<string, number>(); // `${personId}/${photoId}` -> row
  private readonly byPerson = new Map<string, Set<number>>();
  private readonly ordOf = new Map<string, number>();
  private ordPerson: string[] = [];
  private readonly freeOrds: number[] = [];
  private readonly inactive = new Set<string>();

  constructor(
    readonly dim: number,
    initialCapacity = 1024,
  ) {
    this.capacity = Math.max(1, initialCapacity);
    this.data = new Float32Array(this.capacity * dim);
    this.rowOrd = new Int32Array(this.capacity);
  }

  private grow(min: number): void {
    if (min <= this.capacity) return;
    let cap = this.capacity;
    while (cap < min) cap *= 2;
    const d = new Float32Array(cap * this.dim);
    d.set(this.data.subarray(0, this.n * this.dim));
    const o = new Int32Array(cap);
    o.set(this.rowOrd.subarray(0, this.n));
    this.data = d;
    this.rowOrd = o;
    this.capacity = cap;
  }

  private ordinal(personId: string): number {
    let o = this.ordOf.get(personId);
    if (o === undefined) {
      o = this.freeOrds.pop() ?? this.ordPerson.length;
      this.ordPerson[o] = personId;
      this.ordOf.set(personId, o);
    }
    return o;
  }

  upsertPhoto(personId: string, photoId: string, emb: Float32Array): void {
    if (emb.length !== this.dim) throw new Error(`Embedding dim ${emb.length} != ${this.dim}`);
    const key = `${personId}/${photoId}`;
    let row = this.rowOf.get(key);
    if (row === undefined) {
      this.grow(this.n + 1);
      row = this.n++;
      this.rows[row] = { personId, photoId };
      this.rowOf.set(key, row);
      this.rowOrd[row] = this.ordinal(personId);
      let set = this.byPerson.get(personId);
      if (!set) this.byPerson.set(personId, (set = new Set()));
      set.add(row);
    }
    this.data.set(emb, row * this.dim);
  }

  removePhoto(personId: string, photoId: string): void {
    const key = `${personId}/${photoId}`;
    const row = this.rowOf.get(key);
    if (row === undefined) return;
    const last = this.n - 1;
    const set = this.byPerson.get(personId)!;
    set.delete(row);
    this.rowOf.delete(key);
    if (row !== last) {
      // swap-remove: move the last row into the hole
      const moved = this.rows[last];
      this.data.copyWithin(row * this.dim, last * this.dim, (last + 1) * this.dim);
      this.rows[row] = moved;
      this.rowOrd[row] = this.rowOrd[last];
      this.rowOf.set(`${moved.personId}/${moved.photoId}`, row);
      const ms = this.byPerson.get(moved.personId)!;
      ms.delete(last);
      ms.add(row);
    }
    this.rows.length = last;
    this.n = last;
    if (set.size === 0) this.dropPerson(personId);
  }

  private dropPerson(personId: string): void {
    this.byPerson.delete(personId);
    const o = this.ordOf.get(personId);
    if (o !== undefined) {
      this.ordOf.delete(personId);
      this.ordPerson[o] = '';
      this.freeOrds.push(o);
    }
  }

  removePerson(personId: string): void {
    const set = this.byPerson.get(personId);
    if (!set) return;
    for (const row of [...set].sort((a, b) => b - a)) {
      const r = this.rows[row];
      this.removePhoto(r.personId, r.photoId);
    }
    this.inactive.delete(personId);
  }

  setPersonActive(personId: string, active: boolean): void {
    if (active) this.inactive.delete(personId);
    else this.inactive.add(personId);
  }

  has(personId: string): boolean {
    return this.byPerson.has(personId);
  }

  stats(): { persons: number; embeddings: number } {
    return { persons: this.byPerson.size, embeddings: this.n };
  }

  /** Raw cosine scores of `probe` against every row. */
  scoreAll(probe: Float32Array): Float32Array {
    const n = this.n;
    const d = this.dim;
    const data = this.data;
    const out = new Float32Array(n);
    for (let r = 0; r < n; r++) {
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      let s3 = 0;
      let off = r * d;
      let i = 0;
      for (; i + 3 < d; i += 4, off += 4) {
        s0 += data[off] * probe[i];
        s1 += data[off + 1] * probe[i + 1];
        s2 += data[off + 2] * probe[i + 2];
        s3 += data[off + 3] * probe[i + 3];
      }
      for (; i < d; i++, off++) s0 += data[off] * probe[i];
      out[r] = s0 + s1 + s2 + s3;
    }
    return out;
  }

  /**
   * Per-person max score, top-K persons by score. Disabled persons (unless
   * `includeInactive`) and `exclude` are skipped.
   */
  match(probe: Float32Array, topK: number, opts: { exclude?: string; includeInactive?: boolean } = {}): MatchCandidate[] {
    const scores = this.scoreAll(probe);
    const nOrd = this.ordPerson.length;
    const best = new Float32Array(nOrd).fill(-Infinity);
    const bestRow = new Int32Array(nOrd).fill(-1);
    const skip = new Uint8Array(nOrd);
    if (!opts.includeInactive) {
      for (const pid of this.inactive) {
        const o = this.ordOf.get(pid);
        if (o !== undefined) skip[o] = 1;
      }
    }
    if (opts.exclude) {
      const o = this.ordOf.get(opts.exclude);
      if (o !== undefined) skip[o] = 1;
    }
    for (let r = 0; r < this.n; r++) {
      const o = this.rowOrd[r];
      if (skip[o]) continue;
      if (scores[r] > best[o]) {
        best[o] = scores[r];
        bestRow[o] = r;
      }
    }
    const cands: MatchCandidate[] = [];
    for (let o = 0; o < nOrd; o++) {
      if (bestRow[o] < 0) continue;
      cands.push({ personId: this.ordPerson[o], photoId: this.rows[bestRow[o]].photoId, score: best[o] });
    }
    cands.sort((a, b) => b.score - a.score);
    return cands.slice(0, topK);
  }

  /** Max score of `probe` against one person's embeddings (null if the person has none). */
  scorePerson(personId: string, probe: Float32Array): number | null {
    const set = this.byPerson.get(personId);
    if (!set || set.size === 0) return null;
    let best = -Infinity;
    for (const r of set) {
      let s = 0;
      const off = r * this.dim;
      for (let i = 0; i < this.dim; i++) s += this.data[off + i] * probe[i];
      if (s > best) best = s;
    }
    return best;
  }

  personEmbeddings(personId: string): Float32Array[] {
    const set = this.byPerson.get(personId);
    if (!set) return [];
    return [...set].map((r) => this.data.slice(r * this.dim, (r + 1) * this.dim));
  }

  personIds(): string[] {
    return [...this.byPerson.keys()];
  }

  isActive(personId: string): boolean {
    return !this.inactive.has(personId);
  }

  /** Copy of the matrix and row mapping (for the on-disk cache). */
  snapshot(): { matrix: Float32Array; rows: GalleryRow[] } {
    return { matrix: this.data.slice(0, this.n * this.dim), rows: this.rows.slice(0, this.n) };
  }

  clear(): void {
    this.n = 0;
    this.rows = [];
    this.rowOf.clear();
    this.byPerson.clear();
    this.ordOf.clear();
    this.ordPerson = [];
    this.freeOrds.length = 0;
  }

  /** Bulk load (replaces current contents). */
  load(matrix: Float32Array, rows: GalleryRow[]): void {
    if (matrix.length !== rows.length * this.dim) throw new Error('Index cache shape mismatch');
    this.clear();
    this.grow(rows.length);
    this.data.set(matrix);
    this.n = rows.length;
    rows.forEach((r, i) => {
      this.rows[i] = r;
      this.rowOf.set(`${r.personId}/${r.photoId}`, i);
      this.rowOrd[i] = this.ordinal(r.personId);
      let set = this.byPerson.get(r.personId);
      if (!set) this.byPerson.set(r.personId, (set = new Set()));
      set.add(i);
    });
  }
}
