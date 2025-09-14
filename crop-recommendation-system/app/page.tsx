import { CropRecommendationForm } from "@/components/crop-recommendation-form"
import { Header } from "@/components/header"
import { FeatureCards } from "@/components/feature-cards"
import { Footer } from "@/components/footer"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { BarChart3 } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-12">
          {/* Hero Section */}
          <section className="text-center space-y-6">
            <h1 className="text-4xl md:text-6xl font-bold text-foreground text-balance">Smart Crop Recommendations</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto text-pretty">
              Get AI-powered crop recommendations based on your soil conditions and climate data. Make informed
              decisions to maximize your harvest.
            </p>
            <div className="flex justify-center">
              <Link href="/dashboard">
                <Button variant="outline" size="lg" className="gap-2 bg-transparent">
                  <BarChart3 className="h-5 w-5" />
                  View Analytics Dashboard
                </Button>
              </Link>
            </div>
          </section>

          {/* Feature Cards */}
          <FeatureCards />

          {/* Main Form */}
          <section className="space-y-6">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-foreground mb-4">Get Your Crop Recommendation</h2>
              <p className="text-muted-foreground">
                Enter your soil and climate data below to receive personalized crop suggestions
              </p>
            </div>
            <CropRecommendationForm />
          </section>
        </div>
      </main>
      <Footer />
    </div>
  )
}
