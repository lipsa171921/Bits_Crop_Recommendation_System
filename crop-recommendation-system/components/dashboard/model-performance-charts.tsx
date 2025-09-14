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
  LineChart,
  Line,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts"

const modelAccuracyData = [
  { model: "Random Forest", accuracy: 94.2, precision: 93.8, recall: 94.1, f1Score: 93.9 },
  { model: "Gradient Boosting", accuracy: 92.7, precision: 92.3, recall: 92.5, f1Score: 92.4 },
  { model: "SVM", accuracy: 91.5, precision: 91.2, recall: 91.3, f1Score: 91.2 },
  { model: "Neural Network", accuracy: 90.8, precision: 90.5, recall: 90.6, f1Score: 90.5 },
  { model: "Logistic Regression", accuracy: 88.9, precision: 88.6, recall: 88.7, f1Score: 88.6 },
  { model: "Decision Tree", accuracy: 87.3, precision: 87.0, recall: 87.1, f1Score: 87.0 },
  { model: "KNN", accuracy: 85.6, precision: 85.3, recall: 85.4, f1Score: 85.3 },
  { model: "Naive Bayes", accuracy: 83.2, precision: 82.9, recall: 83.0, f1Score: 82.9 },
]

const trainingProgressData = [
  { epoch: 1, trainAccuracy: 65.2, valAccuracy: 63.8 },
  { epoch: 2, trainAccuracy: 72.1, valAccuracy: 70.5 },
  { epoch: 3, trainAccuracy: 78.9, valAccuracy: 76.2 },
  { epoch: 4, trainAccuracy: 84.3, valAccuracy: 81.7 },
  { epoch: 5, trainAccuracy: 88.7, valAccuracy: 85.9 },
  { epoch: 6, trainAccuracy: 91.5, valAccuracy: 88.8 },
  { epoch: 7, trainAccuracy: 93.2, valAccuracy: 90.6 },
  { epoch: 8, trainAccuracy: 94.2, valAccuracy: 91.8 },
  { epoch: 9, trainAccuracy: 94.8, valAccuracy: 92.3 },
  { epoch: 10, trainAccuracy: 95.1, valAccuracy: 92.7 },
]

const modelComparisonRadar = [
  {
    model: "Random Forest",
    accuracy: 94,
    speed: 85,
    interpretability: 75,
    robustness: 90,
    scalability: 80,
  },
  {
    model: "Gradient Boosting",
    accuracy: 93,
    speed: 70,
    interpretability: 65,
    robustness: 85,
    scalability: 75,
  },
  {
    model: "SVM",
    accuracy: 92,
    speed: 60,
    interpretability: 40,
    robustness: 80,
    scalability: 65,
  },
]

export function ModelPerformanceCharts() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Model Accuracy Comparison */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Model Performance Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelAccuracyData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="model" className="text-xs" angle={-45} textAnchor="end" height={80} />
              <YAxis className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
              />
              <Bar dataKey="accuracy" fill="hsl(var(--chart-1))" name="Accuracy %" />
              <Bar dataKey="precision" fill="hsl(var(--chart-2))" name="Precision %" />
              <Bar dataKey="recall" fill="hsl(var(--chart-3))" name="Recall %" />
              <Bar dataKey="f1Score" fill="hsl(var(--chart-4))" name="F1-Score %" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Training Progress */}
      <Card>
        <CardHeader>
          <CardTitle>Training Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={trainingProgressData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="epoch" className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
              />
              <Line
                type="monotone"
                dataKey="trainAccuracy"
                stroke="hsl(var(--chart-1))"
                strokeWidth={2}
                name="Training Accuracy"
              />
              <Line
                type="monotone"
                dataKey="valAccuracy"
                stroke="hsl(var(--chart-2))"
                strokeWidth={2}
                name="Validation Accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Model Characteristics Radar */}
      <Card>
        <CardHeader>
          <CardTitle>Model Characteristics</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={modelComparisonRadar[0]}>
              <PolarGrid className="stroke-muted" />
              <PolarAngleAxis dataKey="subject" className="text-xs" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} className="text-xs" />
              <Radar
                name="Random Forest"
                dataKey="accuracy"
                stroke="hsl(var(--chart-1))"
                fill="hsl(var(--chart-1))"
                fillOpacity={0.1}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
