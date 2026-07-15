import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';

import { Ask } from './ask';

describe('Ask', () => {
  let component: Ask;
  let fixture: ComponentFixture<Ask>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Ask],
      providers: [provideHttpClient()],
    }).compileComponents();

    fixture = TestBed.createComponent(Ask);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
