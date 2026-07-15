import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { marked } from 'marked';

import { AskApi, Source } from './ask-api';

@Component({
  selector: 'app-ask',
  imports: [FormsModule],
  templateUrl: './ask.html',
  styleUrl: './ask.css',
})
export class Ask {
  private readonly askApi = inject(AskApi);

  // Clickable starter prompts shown under the input; each just fills the question signal.
  protected readonly examples = [
    'ما أركان العقد الصحيح؟',
    'اشرح الركن المعنوي في القانون الجنائي',
    'ما الفرق بين الإهمال والقصد في المسؤولية التقصيرية؟',
  ];

  protected readonly question = signal('');
  protected readonly loading = signal(false);
  protected readonly stage = signal<string | null>(null);
  protected readonly streamedText = signal('');
  protected readonly sources = signal<Source[]>([]);
  protected readonly invalidCitations = signal<number[]>([]);
  protected readonly error = signal<string | null>(null);

  // Angular auto-sanitizes [innerHTML] bindings at bind time, so this plain
  // string is safe to render directly in the template without extra handling.
  protected readonly renderedAnswer = computed(() => {
    const text = this.streamedText();
    return text ? (marked.parse(text, { async: false }) as string) : '';
  });

  submit(): void {
    const question = this.question().trim();
    if (!question) {
      return;
    }

    this.loading.set(true);
    this.error.set(null);
    this.stage.set(null);
    this.streamedText.set('');
    this.sources.set([]);
    this.invalidCitations.set([]);

    this.askApi.askStream(question).subscribe({
      next: (event) => {
        switch (event.type) {
          case 'stage':
            this.stage.set(event.stage);
            break;
          case 'token':
            this.stage.set(null);
            this.streamedText.update((current) => current + event.text);
            break;
          case 'refused':
            this.streamedText.set(event.answer);
            break;
          case 'done':
            this.sources.set(event.sources);
            this.invalidCitations.set(event.invalid_citations);
            break;
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
