import { Routes } from '@angular/router';

import { Ask } from './ask/ask';
import { Dashboard } from './dashboard/dashboard';
import { PromptTesting } from './prompt-testing/prompt-testing';
import { Resources } from './resources/resources';

export const routes: Routes = [
  { path: '', redirectTo: 'ask', pathMatch: 'full' },
  { path: 'ask', component: Ask },
  { path: 'dashboard', component: Dashboard },
  { path: 'resources', component: Resources },
  { path: 'testing', component: PromptTesting },
];
