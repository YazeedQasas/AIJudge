import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';

import { ResourcesApi } from './resources-api';

describe('ResourcesApi', () => {
  let service: ResourcesApi;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient()] });
    service = TestBed.inject(ResourcesApi);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
