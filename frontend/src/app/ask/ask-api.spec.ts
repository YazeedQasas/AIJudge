import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';

import { AskApi } from './ask-api';

describe('AskApi', () => {
  let service: AskApi;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient()] });
    service = TestBed.inject(AskApi);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
