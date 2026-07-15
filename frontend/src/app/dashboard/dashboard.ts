import { Component, OnInit, inject, signal } from '@angular/core';

import { DashboardApi, DashboardResponse } from './dashboard-api';

@Component({
  selector: 'app-dashboard',
  imports: [],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css',
})
export class Dashboard implements OnInit {
  private readonly dashboardApi = inject(DashboardApi);

  protected readonly data = signal<DashboardResponse | null>(null);
  protected readonly loading = signal(true);
  protected readonly error = signal<string | null>(null);

  ngOnInit(): void {
    this.dashboardApi.get().subscribe({
      next: (result) => {
        this.data.set(result);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set('تعذّر تحميل بيانات اللوحة. هل الخادم قيد التشغيل؟');
        this.loading.set(false);
        console.error(err);
      },
    });
  }
}
