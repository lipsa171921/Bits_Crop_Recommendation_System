import { type NextRequest, NextResponse } from "next/server"

// Mock prediction function - in production, this would load and use the trained ML models
function mockCropPrediction(soilData: any) {
  const { nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall } = soilData

  // Simple rule-based logic for demonstration
  let primaryCrop = "wheat"
  let confidence = 0.75

  // Rice conditions
  if (rainfall > 200 && humidity > 80) {
    primaryCrop = "rice"
    confidence = 0.85
  }
  // Cotton conditions
  else if (temperature > 30 && rainfall < 100) {
    primaryCrop = "cotton"
    confidence = 0.8
  }
  // Banana conditions
  else if (ph > 7 && potassium > 150) {
    primaryCrop = "banana"
    confidence = 0.78
  }
  // Apple conditions
  else if (temperature < 20 && rainfall > 150) {
    primaryCrop = "apple"
    confidence = 0.82
  }
  // Maize conditions
  else if (nitrogen > 80 && temperature > 25) {
    primaryCrop = "maize"
    confidence = 0.77
  }

  // Generate alternative recommendations
  const allCrops = ["rice", "wheat", "maize", "cotton", "banana", "apple", "mango", "grapes", "chickpea", "lentil"]
  const alternatives = allCrops
    .filter((crop) => crop !== primaryCrop)
    .slice(0, 3)
    .map((crop, index) => ({
      crop,
      probability: confidence - (index + 1) * 0.1,
    }))

  // Soil analysis
  const getNutrientLevel = (value: number, low: number, high: number) => {
    if (value < low) return "Low"
    if (value > high) return "High"
    return "Medium"
  }

  const getPhCategory = (ph: number) => {
    if (ph < 6.0) return "Acidic"
    if (ph > 7.5) return "Alkaline"
    return "Neutral"
  }

  // Generate growing tips based on conditions
  const tips = []

  if (nitrogen < 40) {
    tips.push("Consider adding nitrogen-rich fertilizers or organic compost to improve soil fertility.")
  }
  if (phosphorus < 30) {
    tips.push("Phosphorus levels are low. Add bone meal or rock phosphate to enhance root development.")
  }
  if (potassium < 40) {
    tips.push("Increase potassium levels with wood ash or potassium sulfate for better disease resistance.")
  }
  if (ph < 6.0) {
    tips.push("Soil is acidic. Consider adding lime to raise pH and improve nutrient availability.")
  }
  if (ph > 8.0) {
    tips.push("Soil is alkaline. Add sulfur or organic matter to lower pH.")
  }
  if (rainfall < 50) {
    tips.push("Low rainfall area. Implement drip irrigation or water conservation techniques.")
  }
  if (humidity > 85) {
    tips.push("High humidity may increase disease risk. Ensure good air circulation and drainage.")
  }

  // Add general tips
  tips.push(
    `${primaryCrop.charAt(0).toUpperCase() + primaryCrop.slice(1)} grows best with regular monitoring of soil moisture and nutrients.`,
  )
  tips.push("Consider crop rotation to maintain soil health and prevent pest buildup.")

  return {
    primary_recommendation: primaryCrop,
    confidence,
    top_3_recommendations: [{ crop: primaryCrop, probability: confidence }, ...alternatives],
    soil_analysis: {
      nitrogen_level: getNutrientLevel(nitrogen, 30, 80),
      phosphorus_level: getNutrientLevel(phosphorus, 25, 75),
      potassium_level: getNutrientLevel(potassium, 35, 85),
      ph_category: getPhCategory(ph),
    },
    growing_tips: tips.slice(0, 5), // Limit to 5 tips
  }
}

export async function POST(request: NextRequest) {
  try {
    const soilData = await request.json()

    // Validate input data
    const requiredFields = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
    for (const field of requiredFields) {
      if (typeof soilData[field] !== "number") {
        return NextResponse.json({ error: `Invalid or missing field: ${field}` }, { status: 400 })
      }
    }

    // In production, you would:
    // 1. Load the trained ML model
    // 2. Preprocess the input data (scaling, feature engineering)
    // 3. Make predictions using the model
    // 4. Post-process the results

    // For now, use mock prediction
    const prediction = mockCropPrediction(soilData)

    return NextResponse.json(prediction)
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json({ error: "Failed to process prediction request" }, { status: 500 })
  }
}
