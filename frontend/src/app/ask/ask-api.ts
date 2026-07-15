import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

const API_BASE_URL = 'http://localhost:8001';

export interface Source {
  number: number;
  source: string;
  chunk_index: number;
  score: number;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  invalid_citations: number[];
}

export type AskStreamEvent =
  | { type: 'stage'; stage: string }
  | { type: 'token'; text: string }
  | { type: 'done'; sources: Source[]; invalid_citations: number[] }
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
export class AskApi {
  private readonly http = inject(HttpClient);

  ask(question: string): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${API_BASE_URL}/ask`, { question });
  }

  // Not built on HttpClient: EventSource (the browser's native SSE client) only
  // supports GET with no body, but questions can be long. This uses fetch() with
  // POST and reads the streaming response body by hand instead.
  askStream(question: string): Observable<AskStreamEvent> {
    return new Observable<AskStreamEvent>((subscriber) => {
      const controller = new AbortController();

      fetch(`${API_BASE_URL}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
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

            // {stream: true} tells the decoder to hold onto any trailing partial
            // multi-byte UTF-8 character instead of mangling it -- relevant given
            // our corpus already has real multi-byte characters (em dashes) in it.
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

      // Runs on unsubscribe (component destroyed, or a new question submitted
      // before this stream finished) -- actually cancels the network request,
      // not just our own code's interest in it.
      return () => controller.abort();
    });
  }
}
