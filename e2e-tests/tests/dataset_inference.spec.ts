import { test, expect } from '@playwright/test';


test('Check Dataset Inference works', async ({ page }) => {
    // Navigate to the website
    await page.goto('http://localhost:8501');
  
    // Check that the H1 element with "Dataset Inference" text is not visible
    await expect(page.locator('h1', { hasText: 'Dataset Inference' })).not.toBeVisible();
  
    // Click on the UL element with the text "Dataset Inference"
    await expect(page.locator('ul >> a', { hasText: 'Dataset Inference' })).toBeVisible();
    await page.locator('ul >> a', { hasText: 'Dataset Inference' }).click();
  
    // Wait up to 5 seconds for the H1 element with "Dataset Inference" text to be visible
    await expect(page.locator('h1', { hasText: 'Dataset Inference' })).toBeVisible({ timeout: 5000 });
  
    // Check that the Select element with the options "YOLO Docker" and "YOLO Local" is visible
    await page.screenshot({ path: 'screenshot.png', fullPage: true });
    await expect(page.locator('label', { hasText: 'Select Endpoint' })).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("Yolo Docker")).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("Select model")).not.toBeVisible();
    await expect(page.getByText("Select dataset")).not.toBeVisible();
    await expect(page.getByText("Minimum confidence score")).not.toBeVisible();

    await page.getByText("Yolo Docker").click();
    await page.screenshot({ path: 'screenshot2.png', fullPage: true });
    await page.getByText("MMDET Local").click();
    await page.screenshot({ path: 'screenshot3.png', fullPage: true });
    await expect(page.getByText("Select model")).toBeVisible();
    await expect(page.getByText("Select dataset")).toBeVisible();
    await expect(page.getByText("Minimum confidence score")).toBeVisible();
    await expect(page.getByText(/EmptyDataError/)).not.toBeVisible();

  });