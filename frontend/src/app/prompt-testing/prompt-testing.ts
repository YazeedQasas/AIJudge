import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { marked } from 'marked';

import { PromptInfo, PromptTestingApi, Source } from './prompt-testing-api';

@Component({
  selector: 'app-prompt-testing',
  imports: [FormsModule],
  templateUrl: './prompt-testing.html',
  styleUrl: './prompt-testing.css',
})
export class PromptTesting implements OnInit {
  private readonly api = inject(PromptTestingApi);

  protected readonly versions = signal<PromptInfo[]>([]);
  protected readonly selectedVersion = signal<string>('');
  protected readonly question = signal('');
  protected readonly loading = signal(false);
  protected readonly stage = signal<string | null>(null);
  protected readonly error = signal<string | null>(null);

  protected readonly answer = signal('');
  protected readonly sources = signal<Source[]>([]);
  protected readonly invalidCitations = signal<number[]>([]);
  // Metadata of the version that actually produced the shown verdict (from GET /prompts,
  // matched against the prompt_version the backend stamped on the answer).
  protected readonly usedPrompt = signal<PromptInfo | null>(null);

  protected readonly renderedAnswer = computed(() => {
    const text = this.answer();
    return text ? (marked.parse(text, { async: false }) as string) : '';
  });

  // The metadata of the version currently picked in the dropdown, shown before any
  // verdict is generated so the user can inspect a version by selecting it.
  protected readonly selectedPrompt = computed(
    () => this.versions().find((v) => v.version === this.selectedVersion()) ?? null,
  );

  // Flatten the selected version's generation params into a display-friendly list.
  protected readonly selectedParams = computed(() =>
    Object.entries(this.selectedPrompt()?.generation ?? {}).map(([key, value]) => ({ key, value })),
  );

  // Flatten the used version's generation params into a display-friendly list.
  protected readonly usedParams = computed(() =>
    Object.entries(this.usedPrompt()?.generation ?? {}).map(([key, value]) => ({ key, value })),
  );

  ngOnInit(): void {
    this.api.listPrompts().subscribe({
      next: (versions) => {
        this.versions.set(versions);
        if (versions.length > 0) {
          // Default to the newest version so the picker starts on the latest.
          this.selectedVersion.set(versions[versions.length - 1].version);
        }
      },
      error: (err) => {
        this.error.set('تعذّر تحميل إصدارات النماذج. هل الخادم قيد التشغيل؟');
        console.error(err);
      },
    });
  }

  submit(): void {
    const question = this.question().trim();
    const version = this.selectedVersion();
    if (!question || !version) {
      return;
    }

    this.loading.set(true);
    this.error.set(null);
    this.stage.set(null);
    this.answer.set('');
    this.sources.set([]);
    this.invalidCitations.set([]);
    this.usedPrompt.set(null);

    this.api.askStream(question, version).subscribe({
      next: (event) => {
        switch (event.type) {
          case 'stage':
            this.stage.set(event.stage);
            break;
          case 'token':
            this.stage.set(null);
            this.answer.update((current) => current + event.text);
            break;
          case 'refused':
            this.answer.set(event.answer);
            break;
          case 'done': {
            this.sources.set(event.sources);
            this.invalidCitations.set(event.invalid_citations);
            // Build the card from what the backend ACTUALLY used this run. Only the
            // description is pulled from the fetched list (it's static documentation,
            // not a runtime value, so drift there is harmless).
            const listed = this.versions().find((v) => v.version === event.prompt_version);
            this.usedPrompt.set(
              event.prompt_version
                ? {
                    version: event.prompt_version,
                    name: listed?.name ?? 'judge',
                    model: event.model,
                    description: listed?.description ?? null,
                    generation: event.generation,
                  }
                : null,
            );
            break;
          }
        }
      },
      error: (err) => {
        this.error.set('حدث خطأ أثناء الاتصال بالخادم. هل الخادم قيد التشغيل؟');
        this.loading.set(false);
        console.error(err);
      },
      complete: () => {
        this.loading.set(false);
      },
    });
  }
}
