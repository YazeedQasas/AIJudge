import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';

import { Resources } from './resources';

describe('Resources', () => {
  let component: Resources;
  let fixture: ComponentFixture<Resources>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Resources],
      providers: [provideHttpClient()],
    }).compileComponents();

    fixture = TestBed.createComponent(Resources);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
