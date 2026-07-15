import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

const API_BASE_URL = 'http://localhost:8001';

export interface Chunk {
  id: number;
  chunk_index: number;
  text: string;
}

export interface Document {
  source: string;
  chunks: Chunk[];
}

export interface ResourcesResponse {
  documents: Document[];
}

// Shape of a single hit from the backend's POST /retrieve (semantic vector search).
export interface SearchResult {
  score: number;
  text: string;
  source: string;
  chunk_index: number;
}

interface RetrieveResponse {
  results: SearchResult[];
}

@Injectable({
  providedIn: 'root',
})
export class ResourcesApi {
  private readonly http = inject(HttpClient);

  list(): Observable<ResourcesResponse> {
    return this.http.get<ResourcesResponse>(`${API_BASE_URL}/resources`);
  }

  // Semantic search: embeds the query and cosine-searches the vector store,
  // reusing the same /retrieve endpoint the Ask flow uses under the hood.
  search(query: string, limit = 10): Observable<RetrieveResponse> {
    return this.http.post<RetrieveResponse>(`${API_BASE_URL}/retrieve`, { query, limit });
  }
}
