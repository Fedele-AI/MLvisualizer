/* tslint:disable */
/* eslint-disable */
export enum InstrumentType {
  Robo = 0,
  Piano = 1,
  Guitar = 2,
}
export class MusicTransformer {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  /**
   * Set melodic mode
   */
  set_melodic(melodic: boolean): void;
  /**
   * Set random spacing mode
   */
  set_random_spacing(random_spacing: boolean): void;
  /**
   * Set the instrument type
   */
  set_instrument(instrument: InstrumentType): void;
  /**
   * Set the tempo in beats per minute
   */
  set_tempo(bpm: number): void;
  /**
   * Set the sample rate
   */
  set_sample_rate(rate: number): void;
  /**
   * Set the target duration
   */
  set_target_duration(duration: number): void;
  /**
   * Get the sample rate
   */
  get_sample_rate(): number;
  /**
   * Get the target duration
   */
  get_target_duration(): number;
  /**
   * Generate a new musical sequence using transformer-like attention
   */
  generate_sequence(): Uint32Array;
  /**
   * Generate audio samples for the current sequence
   */
  generate_audio(): Float32Array;
  /**
   * Get the duration of the current sequence in seconds
   */
  get_duration(): number;
  /**
   * Get the current sequence
   */
  get_sequence(): Uint32Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_musictransformer_free: (a: number, b: number) => void;
  readonly musictransformer_new: () => number;
  readonly musictransformer_set_melodic: (a: number, b: number) => void;
  readonly musictransformer_set_random_spacing: (a: number, b: number) => void;
  readonly musictransformer_set_instrument: (a: number, b: number) => void;
  readonly musictransformer_set_tempo: (a: number, b: number) => void;
  readonly musictransformer_set_sample_rate: (a: number, b: number) => void;
  readonly musictransformer_set_target_duration: (a: number, b: number) => void;
  readonly musictransformer_get_sample_rate: (a: number) => number;
  readonly musictransformer_get_target_duration: (a: number) => number;
  readonly musictransformer_generate_sequence: (a: number) => [number, number];
  readonly musictransformer_generate_audio: (a: number) => [number, number];
  readonly musictransformer_get_duration: (a: number) => number;
  readonly musictransformer_get_sequence: (a: number) => [number, number];
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
