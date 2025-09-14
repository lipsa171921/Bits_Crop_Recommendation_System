"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Cell,
  PieChart,
  Pie,
} from "recharts"

const soilParameterDistribution = [
  { parameter: "Nitrogen", low: 15, medium: 65, high: 20 },
  { parameter: "Phosphorus", low: 25, medium: 55, high: 20 },
  { parameter: "Potassium", low: 20, medium: 60, high: 20 },
  { parameter: "pH", acidic: 30, neutral: 50, alkaline: 20 },
]

const climateData = [
  { month: "Jan", temperature: 18.2, humidity: 72, rainfall: 45 },
  { month: "Feb", temperature: 20.1, humidity: 68, rainfall: 38 },
  { month: "Mar", temperature: 23.5, humidity: 65, rainfall: 52 },
  { month: "Apr", temperature: 26.8, humidity: 70, rainfall: 78 },
  { month: "May", temperature: 29.3, humidity: 75, rainfall: 125 },
  { month: "Jun", temperature: 31.2, humidity: 82, rainfall: 185 },
  { month: "Jul", temperature: 30.8, humidity: 85, rainfall: 220 },
  { month: "Aug", temperature: 30.1, humidity: 83, rainfall: 195 },
  { month: "Sep", temperature: 28.4, humidity: 78, rainfall: 145 },
  { month: "Oct", temperature: 25.6, humidity: 74, rainfall: 85 },
  { month: "Nov", temperature: 22.1, humidity: 71, rainfall: 55 },
  { month: "Dec", temperature: 19.3, humidity: 73, rainfall: 42 },
]

const temperatureHumidityScatter = [
  { temperature: 15.2, humidity: 85, crop: "apple" },
  { temperature: 18.5, humidity: 75, crop: "wheat" },
  { temperature: 22.1, humidity: 70, crop: "maize" },
  { temperature: 25.8, humidity: 65, crop: "cotton" },
  { temperature: 28.3, humidity: 80, crop: "rice" },
  { temperature: 30.1, humidity: 75, crop: "banana" },
  { temperature: 32.5, humidity: 60, crop: "mango" },
  { temperature: 35.2, humidity: 55, crop: "cotton" },
]

const cropSeasonality = [
  { name: "Kharif (Monsoon)", value: 45, fill: "hsl(var(--chart-1))" },
  { name: "Rabi (Winter)", value: 35, fill: "hsl(var(--chart-2))" },
  { name: "Zaid (Summer)", value: 20, fill: "hsl(var(--chart-3))" },
]

const COLORS = ["hsl(var(--chart-1))", "hsl(var(--chart-2))", "hsl(var(--chart-3))", "hsl(var(--chart-4))"]

export function DataAnalyticsCharts() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Climate Trends */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Climate Parameter Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={climateData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="month" className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
              />
              <Area
                type="monotone"
                dataKey="temperature"
                stackId="1"
                stroke="hsl(var(--chart-1))"
                fill="hsl(var(--chart-1))"
                fillOpacity={0.6}
                name="Temperature (°C)"
              />
              <Area
                type="monotone"
                dataKey="rainfall"
                stackId="2"
                stroke="hsl(var(--chart-2))"
                fill="hsl(var(--chart-2))"
                fillOpacity={0.6}
                name="Rainfall (mm)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Temperature vs Humidity Scatter */}
      <Card>
        <CardHeader>
          <CardTitle>Temperature vs Humidity by Crop</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart data={temperatureHumidityScatter}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis type="number" dataKey="temperature" name="Temperature" unit="°C" className="text-xs" />
              <YAxis type="number" dataKey="humidity" name="Humidity" unit="%" className="text-xs" />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
                formatter={(value, name) => [value, name === "temperature" ? "Temperature (°C)" : "Humidity (%)"]}
              />
              <Scatter name="Crops" dataKey="humidity" fill="hsl(var(--chart-1))">
                {temperatureHumidityScatter.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Crop Seasonality */}
      <Card>
        <CardHeader>
          <CardTitle>Crop Seasonality Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={cropSeasonality}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {cropSeasonality.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
