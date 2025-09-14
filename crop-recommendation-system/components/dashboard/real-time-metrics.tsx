import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, Users, Target, Clock } from "lucide-react"

const metrics = [
  {
    title: "Total Predictions",
    value: "12,847",
    change: "+23%",
    trend: "up",
    icon: Target,
    description: "Crop recommendations made",
  },
  {
    title: "Active Users",
    value: "2,341",
    change: "+12%",
    trend: "up",
    icon: Users,
    description: "Farmers using the system",
  },
  {
    title: "Model Accuracy",
    value: "94.2%",
    change: "+2.1%",
    trend: "up",
    icon: TrendingUp,
    description: "Average across all models",
  },
  {
    title: "Response Time",
    value: "1.2s",
    change: "-0.3s",
    trend: "up",
    icon: Clock,
    description: "Average prediction time",
  },
]

export function RealTimeMetrics() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric, index) => (
        <Card key={index} className="border-border/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">{metric.title}</CardTitle>
            <metric.icon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">{metric.value}</div>
            <div className="flex items-center gap-2 mt-1">
              <span className={`text-sm font-medium ${metric.trend === "up" ? "text-green-600" : "text-red-600"}`}>
                {metric.change}
              </span>
              <span className="text-sm text-muted-foreground">from last month</span>
            </div>
            <p className="text-xs text-muted-foreground mt-2">{metric.description}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
