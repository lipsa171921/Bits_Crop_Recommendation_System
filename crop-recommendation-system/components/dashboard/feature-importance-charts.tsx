"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
} from "recharts"

const featureImportanceData = [
  { feature: "Rainfall", importance: 0.28, model: "Random Forest" },
  { feature: "Temperature", importance: 0.22, model: "Random Forest" },
  { feature: "Humidity", importance: 0.18, model: "Random Forest" },
  { feature: "Potassium", importance: 0.12, model: "Random Forest" },
  { feature: "Nitrogen", importance: 0.1, model: "Random Forest" },
  { feature: "pH", importance: 0.08, model: "Random Forest" },
  { feature: "Phosphorus", importance: 0.02, model: "Random Forest" },
]

const crossModelImportance = [
  { feature: "Rainfall", randomForest: 0.28, gradientBoosting: 0.25, svm: 0.22 },
  { feature: "Temperature", randomForest: 0.22, gradientBoosting: 0.24, svm: 0.26 },
  { feature: "Humidity", randomForest: 0.18, gradientBoosting: 0.2, svm: 0.19 },
  { feature: "Potassium", randomForest: 0.12, gradientBoosting: 0.11, svm: 0.13 },
  { feature: "Nitrogen", randomForest: 0.1, gradientBoosting: 0.12, svm: 0.09 },
  { feature: "pH", randomForest: 0.08, gradientBoosting: 0.06, svm: 0.08 },
  { feature: "Phosphorus", randomForest: 0.02, gradientBoosting: 0.02, svm: 0.03 },
]

const radialImportanceData = [
  { name: "Rainfall", value: 28, fill: "hsl(var(--chart-1))" },
  { name: "Temperature", value: 22, fill: "hsl(var(--chart-2))" },
  { name: "Humidity", value: 18, fill: "hsl(var(--chart-3))" },
  { name: "Potassium", value: 12, fill: "hsl(var(--chart-4))" },
  { name: "Nitrogen", value: 10, fill: "hsl(var(--chart-5))" },
  { name: "pH", value: 8, fill: "hsl(var(--chart-1))" },
  { name: "Phosphorus", value: 2, fill: "hsl(var(--chart-2))" },
]

export function FeatureImportanceCharts() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Feature Importance Bar Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance (Random Forest)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureImportanceData} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis type="number" className="text-xs" />
              <YAxis dataKey="feature" type="category" className="text-xs" width={80} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Importance"]}
              />
              <Bar dataKey="importance" fill="hsl(var(--chart-1))" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Radial Feature Importance */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <RadialBarChart cx="50%" cy="50%" innerRadius="20%" outerRadius="80%" data={radialImportanceData}>
              <RadialBar
                minAngle={15}
                label={{ position: "insideStart", fill: "#fff" }}
                background
                clockWise
                dataKey="value"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                formatter={(value) => [`${value}%`, "Importance"]}
              />
            </RadialBarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Cross-Model Feature Comparison */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Feature Importance Across Models</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={crossModelImportance}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="feature" className="text-xs" angle={-45} textAnchor="end" height={80} />
              <YAxis className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Importance"]}
              />
              <Bar dataKey="randomForest" fill="hsl(var(--chart-1))" name="Random Forest" />
              <Bar dataKey="gradientBoosting" fill="hsl(var(--chart-2))" name="Gradient Boosting" />
              <Bar dataKey="svm" fill="hsl(var(--chart-3))" name="SVM" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
