import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

const API_BASE_URL = 'http://localhost:8000';

export interface CorpusStats {
  document_count: number;
  chunk_count: number;
}

export interface HealthStatus {
  qdrant: boolean;
  lm_studio: boolean;
}

export interface DashboardResponse {
  corpus: CorpusStats;
  health: HealthStatus;
}

@Injectable({
  providedIn: 'root',
})
export class DashboardApi {
  private readonly http = inject(HttpClient);

  get(): Observable<DashboardResponse> {
    return this.http.get<DashboardResponse>(`${API_BASE_URL}/dashboard`);
  }
}
