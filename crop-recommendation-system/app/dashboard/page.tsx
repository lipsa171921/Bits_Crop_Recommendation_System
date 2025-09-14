import { DashboardHeader } from "@/components/dashboard/dashboard-header"
import { ModelPerformanceCharts } from "@/components/dashboard/model-performance-charts"
import { DataAnalyticsCharts } from "@/components/dashboard/data-analytics-charts"
import { FeatureImportanceCharts } from "@/components/dashboard/feature-importance-charts"
import { CropDistributionCharts } from "@/components/dashboard/crop-distribution-charts"
import { RealTimeMetrics } from "@/components/dashboard/real-time-metrics"

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Real-time Metrics */}
          <RealTimeMetrics />

          {/* Model Performance */}
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Model Performance Analysis</h2>
              <p className="text-muted-foreground">
                Compare accuracy, precision, and performance metrics across different ML models
              </p>
            </div>
            <ModelPerformanceCharts />
          </section>

          {/* Data Analytics */}
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Dataset Analytics</h2>
              <p className="text-muted-foreground">
                Explore soil parameter distributions and correlations in the training data
              </p>
            </div>
            <DataAnalyticsCharts />
          </section>

          {/* Feature Importance */}
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Feature Importance</h2>
              <p className="text-muted-foreground">
                Understand which soil and climate factors most influence crop recommendations
              </p>
            </div>
            <FeatureImportanceCharts />
          </section>

          {/* Crop Distribution */}
          <section className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Crop Recommendations Analysis</h2>
              <p className="text-muted-foreground">Analyze crop recommendation patterns and regional suitability</p>
            </div>
            <CropDistributionCharts />
          </section>
        </div>
      </main>
    </div>
  )
}
