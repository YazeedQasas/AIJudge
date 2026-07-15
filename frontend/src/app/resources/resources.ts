import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { Document, ResourcesApi, SearchResult } from './resources-api';

@Component({
  selector: 'app-resources',
  imports: [FormsModule],
  templateUrl: './resources.html',
  styleUrl: './resources.css',
})
export class Resources implements OnInit {
  private readonly resourcesApi = inject(ResourcesApi);

  protected readonly documents = signal<Document[]>([]);
  protected readonly loading = signal(true);
  protected readonly error = signal<string | null>(null);
  protected readonly expandedSources = signal<Set<string>>(new Set());

  // Semantic search state.
  protected readonly query = signal('');
  protected readonly searching = signal(false);
  protected readonly searched = signal(false);
  protected readonly results = signal<SearchResult[]>([]);
  protected readonly searchError = signal<string | null>(null);

  ngOnInit(): void {
    this.resourcesApi.list().subscribe({
      next: (result) => {
        this.documents.set(result.documents);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set('تعذّر تحميل الوثائق. هل الخادم قيد التشغيل؟');
        this.loading.set(false);
        console.error(err);
      },
    });
  }

  search(): void {
    const query = this.query().trim();
    if (!query) {
      return;
    }

    this.searching.set(true);
    this.searchError.set(null);

    this.resourcesApi.search(query).subscribe({
      next: (response) => {
        this.results.set(response.results);
        this.searched.set(true);
        this.searching.set(false);
      },
      error: (err) => {
        this.searchError.set('فشل البحث. هل الخادم قيد التشغيل؟');
        this.searching.set(false);
        console.error(err);
      },
    });
  }

  clearSearch(): void {
    this.query.set('');
    this.results.set([]);
    this.searched.set(false);
    this.searchError.set(null);
  }

  toggle(source: string): void {
    // Signals compare by reference, so mutating the existing Set in place
    // wouldn't be detected as a change — build a new Set each time instead.
    this.expandedSources.update((current) => {
      const next = new Set(current);
      next.has(source) ? next.delete(source) : next.add(source);
      return next;
    });
  }

  isExpanded(source: string): boolean {
    return this.expandedSources().has(source);
  }
}
