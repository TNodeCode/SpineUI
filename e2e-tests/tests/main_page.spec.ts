import { test, expect } from '@playwright/test';

test('site loads and the sidebar is visible', async ({ page }) => {
    await page.goto('http://localhost:8501');
  
    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/App · Streamlit/);

    // Check if the H1 element with the expected text is present
    await expect(page.locator('h1', { hasText: 'Spine Detection App' })).toBeVisible();

    // Check if all the sidebar element are visibe
    await expect(page.locator('ul >> a', { hasText: 'App' })).toBeVisible();
    await expect(page.locator('ul >> a', { hasText: 'Dataset Inference' })).toBeVisible();
    await expect(page.locator('ul >> a', { hasText: 'Dataset Inspection' })).toBeVisible();
    await expect(page.locator('ul >> a', { hasText: 'Datasets' })).toBeVisible();
    await expect(page.locator('ul >> a', { hasText: 'Model Comparison' })).toBeVisible();
    await expect(page.locator('ul >> a', { hasText: 'Model Inference' })).toBeVisible();
    await expect(page.locator('ul >> a', { hasText: 'Tracking' })).toBeVisible();
  });
  