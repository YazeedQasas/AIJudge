import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { HttpClient } from '@angular/common/http';
import { inject } from '@angular/core';

const API_BASE_URL = 'http://localhost:8000';

// One prompt version's public metadata, from GET /prompts. Drives the version
// picker and the "generation info" card shown under a verdict.
export interface PromptInfo {
  version: string;
  name: string;
  model: string | null;
  description: string | null;
  generation: Record<string, number>;
}

export interface Source {
  number: number;
  source: string;
  chunk_index: number;
  score: number;
}

// Streaming events from POST /ask/stream. The 'done' event carries prompt_version
// (which version produced the answer) in addition to sources + invalid citations.
export type AskStreamEvent =
  | { type: 'stage'; stage: string }
  | { type: 'token'; text: string }
  | {
      type: 'done';
      sources: Source[];
      invalid_citations: number[];
      prompt_version: string | null;
      // The exact model + params the backend used for this generation (authoritative,
      // so the "produced by" card can't show a stale value from an edited file).
      model: string | null;
      generation: Record<string, number>;
    }
  | { type: 'refused'; answer: string };

/** Parse one raw SSE event block ("event: X\ndata: Y") into a typed AskStreamEvent. */
function parseSseEvent(raw: string): AskStreamEvent | null {
  let eventType = '';
  let data = '';
  for (const line of raw.split('\n')) {
    if (line.startsWith('event: ')) {
      eventType = line.slice('event: '.length);
    } else if (line.startsWith('data: ')) {
      data = line.slice('data: '.length);
    }
  }
  if (!eventType || !data) {
    return null;
  }
  return { type: eventType, ...JSON.parse(data) } as AskStreamEvent;
}

@Injectable({
  providedIn: 'root',
})
export class PromptTestingApi {
  private readonly http = inject(HttpClient);

  listPrompts(): Observable<PromptInfo[]> {
    return this.http.get<PromptInfo[]>(`${API_BASE_URL}/prompts`);
  }

  // Streams the verdict for a specific prompt version. Same fetch-based SSE reader as
  // the Ask page (EventSource can't POST a body), but sends prompt_version so the
  // backend runs the chosen version and stamps it on the 'done' event.
  askStream(question: string, promptVersion: string): Observable<AskStreamEvent> {
    return new Observable<AskStreamEvent>((subscriber) => {
      const controller = new AbortController();

      fetch(`${API_BASE_URL}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, prompt_version: promptVersion }),
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok || !response.body) {
            throw new Error(`Request failed: ${response.status}`);
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          for (;;) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }

            buffer += decoder.decode(value, { stream: true });

            const events = buffer.split('\n\n');
            buffer = events.pop() ?? ''; // last piece may be incomplete; keep for next read

            for (const rawEvent of events) {
              const parsed = parseSseEvent(rawEvent);
              if (parsed) {
                subscriber.next(parsed);
              }
            }
          }

          subscriber.complete();
        })
        .catch((err) => subscriber.error(err));

      return () => controller.abort();
    });
  }
}
