import { Brain, Leaf, TrendingUp, Shield } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

const features = [
  {
    icon: Brain,
    title: "AI-Powered Analysis",
    description:
      "Advanced machine learning models analyze your soil and climate data to provide accurate recommendations.",
  },
  {
    icon: Leaf,
    title: "Sustainable Farming",
    description: "Get recommendations that promote sustainable agricultural practices and soil health.",
  },
  {
    icon: TrendingUp,
    title: "Maximize Yield",
    description: "Optimize your crop selection to achieve the best possible harvest for your conditions.",
  },
  {
    icon: Shield,
    title: "Risk Assessment",
    description: "Understand potential risks and get alternative crop suggestions for better planning.",
  },
]

export function FeatureCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {features.map((feature, index) => (
        <Card key={index} className="border-border/50 hover:border-primary/50 transition-colors">
          <CardContent className="p-6 text-center space-y-4">
            <div className="mx-auto w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
              <feature.icon className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-semibold text-foreground">{feature.title}</h3>
            <p className="text-sm text-muted-foreground text-pretty">{feature.description}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
