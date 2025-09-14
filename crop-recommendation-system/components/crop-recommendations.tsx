import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, TrendingUp, Droplets, Leaf } from "lucide-react"

interface RecommendationData {
  primary_recommendation: string
  confidence: number
  top_3_recommendations: Array<{
    crop: string
    probability: number
  }>
  soil_analysis: {
    nitrogen_level: string
    phosphorus_level: string
    potassium_level: string
    ph_category: string
  }
  growing_tips: string[]
}

interface CropRecommendationsProps {
  data: RecommendationData
}

const cropEmojis: { [key: string]: string } = {
  rice: "🌾",
  wheat: "🌾",
  maize: "🌽",
  corn: "🌽",
  cotton: "🌿",
  sugarcane: "🎋",
  banana: "🍌",
  mango: "🥭",
  apple: "🍎",
  orange: "🍊",
  grapes: "🍇",
  coconut: "🥥",
  coffee: "☕",
  tea: "🍃",
  potato: "🥔",
  tomato: "🍅",
  onion: "🧅",
  chickpea: "🫘",
  lentil: "🫘",
  soybean: "🫘",
  default: "🌱",
}

const getCropEmoji = (crop: string): string => {
  const lowerCrop = crop.toLowerCase()
  return cropEmojis[lowerCrop] || cropEmojis.default
}

const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return "bg-green-100 text-green-800 border-green-200"
  if (confidence >= 0.6) return "bg-yellow-100 text-yellow-800 border-yellow-200"
  return "bg-red-100 text-red-800 border-red-200"
}

const getLevelColor = (level: string): string => {
  switch (level.toLowerCase()) {
    case "high":
      return "bg-green-100 text-green-800"
    case "medium":
    case "moderate":
      return "bg-yellow-100 text-yellow-800"
    case "low":
      return "bg-red-100 text-red-800"
    default:
      return "bg-gray-100 text-gray-800"
  }
}

export function CropRecommendations({ data }: CropRecommendationsProps) {
  return (
    <div className="space-y-6">
      {/* Primary Recommendation */}
      <Card className="border-primary/20 bg-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-primary rounded-full">
              <CheckCircle className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h3 className="text-xl">Recommended Crop</h3>
              <p className="text-sm text-muted-foreground font-normal">Best match for your conditions</p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <span className="text-4xl">{getCropEmoji(data.primary_recommendation)}</span>
              <div>
                <h4 className="text-2xl font-bold capitalize text-foreground">{data.primary_recommendation}</h4>
                <Badge className={getConfidenceColor(data.confidence)}>
                  {Math.round(data.confidence * 100)}% Confidence
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Alternative Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Alternative Options
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {data.top_3_recommendations.map((rec, index) => (
              <div key={index} className="p-4 border rounded-lg hover:border-primary/50 transition-colors">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">{getCropEmoji(rec.crop)}</span>
                  <h5 className="font-semibold capitalize">{rec.crop}</h5>
                </div>
                <Badge variant="outline">{Math.round(rec.probability * 100)}% Match</Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Soil Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Leaf className="h-5 w-5 text-primary" />
            Soil Analysis Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 border rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Nitrogen</div>
              <Badge className={getLevelColor(data.soil_analysis.nitrogen_level)}>
                {data.soil_analysis.nitrogen_level}
              </Badge>
            </div>
            <div className="text-center p-3 border rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Phosphorus</div>
              <Badge className={getLevelColor(data.soil_analysis.phosphorus_level)}>
                {data.soil_analysis.phosphorus_level}
              </Badge>
            </div>
            <div className="text-center p-3 border rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Potassium</div>
              <Badge className={getLevelColor(data.soil_analysis.potassium_level)}>
                {data.soil_analysis.potassium_level}
              </Badge>
            </div>
            <div className="text-center p-3 border rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">pH Level</div>
              <Badge className={getLevelColor(data.soil_analysis.ph_category)}>{data.soil_analysis.ph_category}</Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Growing Tips */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Droplets className="h-5 w-5 text-primary" />
            Growing Tips
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-3">
            {data.growing_tips.map((tip, index) => (
              <li key={index} className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                <span className="text-foreground">{tip}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
