"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
} from "recharts"

const cropDistributionData = [
  { name: "Rice", value: 18.5, fill: "hsl(var(--chart-1))" },
  { name: "Wheat", value: 16.2, fill: "hsl(var(--chart-2))" },
  { name: "Maize", value: 14.8, fill: "hsl(var(--chart-3))" },
  { name: "Cotton", value: 12.3, fill: "hsl(var(--chart-4))" },
  { name: "Banana", value: 9.7, fill: "hsl(var(--chart-5))" },
  { name: "Mango", value: 8.4, fill: "hsl(var(--chart-1))" },
  { name: "Apple", value: 6.9, fill: "hsl(var(--chart-2))" },
  { name: "Others", value: 13.2, fill: "hsl(var(--chart-3))" },
]

const regionalCropData = [
  { region: "North", rice: 25, wheat: 35, maize: 20, cotton: 10, others: 10 },
  { region: "South", rice: 40, wheat: 5, maize: 15, cotton: 15, others: 25 },
  { region: "East", rice: 45, wheat: 15, maize: 20, cotton: 5, others: 15 },
  { region: "West", rice: 15, wheat: 25, maize: 10, cotton: 30, others: 20 },
  { region: "Central", rice: 30, wheat: 25, maize: 25, cotton: 10, others: 10 },
]

const seasonalTrendsData = [
  { month: "Jan", predictions: 1250, accuracy: 92.3 },
  { month: "Feb", predictions: 1180, accuracy: 93.1 },
  { month: "Mar", predictions: 1420, accuracy: 94.2 },
  { month: "Apr", predictions: 1680, accuracy: 93.8 },
  { month: "May", predictions: 1890, accuracy: 94.5 },
  { month: "Jun", predictions: 2100, accuracy: 95.1 },
  { month: "Jul", predictions: 1950, accuracy: 94.8 },
  { month: "Aug", predictions: 1780, accuracy: 94.2 },
  { month: "Sep", predictions: 1650, accuracy: 93.9 },
  { month: "Oct", predictions: 1520, accuracy: 93.5 },
  { month: "Nov", predictions: 1380, accuracy: 92.8 },
  { month: "Dec", predictions: 1290, accuracy: 92.1 },
]

const cropSuccessRates = [
  { crop: "Rice", successRate: 96.2, totalPredictions: 2847 },
  { crop: "Wheat", successRate: 94.8, totalPredictions: 2156 },
  { crop: "Maize", successRate: 93.5, totalPredictions: 1923 },
  { crop: "Cotton", successRate: 91.7, totalPredictions: 1654 },
  { crop: "Banana", successRate: 89.3, totalPredictions: 1287 },
  { crop: "Mango", successRate: 87.9, totalPredictions: 1098 },
  { crop: "Apple", successRate: 85.4, totalPredictions: 876 },
  { crop: "Grapes", successRate: 83.2, totalPredictions: 654 },
]

export function CropDistributionCharts() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Crop Distribution Pie Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Crop Recommendation Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={cropDistributionData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {cropDistributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                formatter={(value) => [`${value}%`, "Distribution"]}
              />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Crop Success Rates */}
      <Card>
        <CardHeader>
          <CardTitle>Crop Prediction Success Rates</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={cropSuccessRates} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis type="number" domain={[80, 100]} className="text-xs" />
              <YAxis dataKey="crop" type="category" className="text-xs" width={60} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                formatter={(value, name) => [
                  name === "successRate" ? `${value}%` : value,
                  name === "successRate" ? "Success Rate" : "Total Predictions",
                ]}
              />
              <Bar dataKey="successRate" fill="hsl(var(--chart-1))" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Regional Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Regional Crop Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={regionalCropData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="region" className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                formatter={(value) => [`${value}%`, "Distribution"]}
              />
              <Bar dataKey="rice" stackId="a" fill="hsl(var(--chart-1))" name="Rice" />
              <Bar dataKey="wheat" stackId="a" fill="hsl(var(--chart-2))" name="Wheat" />
              <Bar dataKey="maize" stackId="a" fill="hsl(var(--chart-3))" name="Maize" />
              <Bar dataKey="cotton" stackId="a" fill="hsl(var(--chart-4))" name="Cotton" />
              <Bar dataKey="others" stackId="a" fill="hsl(var(--chart-5))" name="Others" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Seasonal Trends */}
      <Card>
        <CardHeader>
          <CardTitle>Monthly Prediction Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={seasonalTrendsData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="month" className="text-xs" />
              <YAxis yAxisId="left" className="text-xs" />
              <YAxis yAxisId="right" orientation="right" className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
              />
              <Bar yAxisId="left" dataKey="predictions" fill="hsl(var(--chart-1))" name="Predictions" />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="accuracy"
                stroke="hsl(var(--chart-2))"
                strokeWidth={3}
                name="Accuracy %"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
