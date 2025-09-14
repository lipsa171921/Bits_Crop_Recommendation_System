"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Info, Loader2, Sprout } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { CropRecommendations } from "./crop-recommendations"

interface SoilData {
  nitrogen: number
  phosphorus: number
  potassium: number
  temperature: number
  humidity: number
  ph: number
  rainfall: number
}

const parameterInfo = {
  nitrogen: "Nitrogen (N) is essential for leaf growth and chlorophyll production. Typical range: 0-140 kg/ha",
  phosphorus: "Phosphorus (P) promotes root development and flowering. Typical range: 5-145 kg/ha",
  potassium: "Potassium (K) helps with disease resistance and water regulation. Typical range: 5-205 kg/ha",
  temperature: "Average temperature during growing season. Typical range: 8-44°C",
  humidity: "Relative humidity percentage. Typical range: 14-100%",
  ph: "Soil pH level affects nutrient availability. Typical range: 3.5-9.9",
  rainfall: "Annual rainfall in millimeters. Typical range: 20-300mm",
}

export function CropRecommendationForm() {
  const [soilData, setSoilData] = useState<SoilData>({
    nitrogen: 50,
    phosphorus: 50,
    potassium: 50,
    temperature: 25,
    humidity: 65,
    ph: 6.5,
    rainfall: 150,
  })

  const [isLoading, setIsLoading] = useState(false)
  const [recommendations, setRecommendations] = useState(null)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    try {
      const response = await fetch("/api/predict-crop", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(soilData),
      })

      if (!response.ok) {
        throw new Error("Failed to get recommendations")
      }

      const data = await response.json()
      setRecommendations(data)
    } catch (err) {
      setError("Failed to get crop recommendations. Please try again.")
      console.error("Error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const updateValue = (key: keyof SoilData, value: number) => {
    setSoilData((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <TooltipProvider>
      <div className="space-y-8">
        <Card className="border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sprout className="h-5 w-5 text-primary" />
              Soil & Climate Parameters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Nitrogen */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="nitrogen">Nitrogen (N) - kg/ha</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{parameterInfo.nitrogen}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[soilData.nitrogen]}
                      onValueChange={(value) => updateValue("nitrogen", value[0])}
                      max={140}
                      min={0}
                      step={1}
                      className="w-full"
                    />
                    <Input
                      id="nitrogen"
                      type="number"
                      value={soilData.nitrogen}
                      onChange={(e) => updateValue("nitrogen", Number(e.target.value))}
                      min={0}
                      max={140}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Phosphorus */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="phosphorus">Phosphorus (P) - kg/ha</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{parameterInfo.phosphorus}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[soilData.phosphorus]}
                      onValueChange={(value) => updateValue("phosphorus", value[0])}
                      max={145}
                      min={5}
                      step={1}
                      className="w-full"
                    />
                    <Input
                      id="phosphorus"
                      type="number"
                      value={soilData.phosphorus}
                      onChange={(e) => updateValue("phosphorus", Number(e.target.value))}
                      min={5}
                      max={145}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Potassium */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="potassium">Potassium (K) - kg/ha</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{parameterInfo.potassium}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[soilData.potassium]}
                      onValueChange={(value) => updateValue("potassium", value[0])}
                      max={205}
                      min={5}
                      step={1}
                      className="w-full"
                    />
                    <Input
                      id="potassium"
                      type="number"
                      value={soilData.potassium}
                      onChange={(e) => updateValue("potassium", Number(e.target.value))}
                      min={5}
                      max={205}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Temperature */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="temperature">Temperature - °C</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{parameterInfo.temperature}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[soilData.temperature]}
                      onValueChange={(value) => updateValue("temperature", value[0])}
                      max={44}
                      min={8}
                      step={0.1}
                      className="w-full"
                    />
                    <Input
                      id="temperature"
                      type="number"
                      value={soilData.temperature}
                      onChange={(e) => updateValue("temperature", Number(e.target.value))}
                      min={8}
                      max={44}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Humidity */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="humidity">Humidity - %</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{parameterInfo.humidity}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[soilData.humidity]}
                      onValueChange={(value) => updateValue("humidity", value[0])}
                      max={100}
                      min={14}
                      step={1}
                      className="w-full"
                    />
                    <Input
                      id="humidity"
                      type="number"
                      value={soilData.humidity}
                      onChange={(e) => updateValue("humidity", Number(e.target.value))}
                      min={14}
                      max={100}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* pH */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="ph">Soil pH</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{parameterInfo.ph}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[soilData.ph]}
                      onValueChange={(value) => updateValue("ph", value[0])}
                      max={9.9}
                      min={3.5}
                      step={0.1}
                      className="w-full"
                    />
                    <Input
                      id="ph"
                      type="number"
                      value={soilData.ph}
                      onChange={(e) => updateValue("ph", Number(e.target.value))}
                      min={3.5}
                      max={9.9}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Rainfall */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="rainfall">Rainfall - mm</Label>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{parameterInfo.rainfall}</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2">
                    <Slider
                      value={[soilData.rainfall]}
                      onValueChange={(value) => updateValue("rainfall", value[0])}
                      max={300}
                      min={20}
                      step={1}
                      className="w-full"
                    />
                    <Input
                      id="rainfall"
                      type="number"
                      value={soilData.rainfall}
                      onChange={(e) => updateValue("rainfall", Number(e.target.value))}
                      min={20}
                      max={300}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {error && (
                <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <p className="text-destructive text-sm">{error}</p>
                </div>
              )}

              <Button type="submit" className="w-full" size="lg" disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing Your Data...
                  </>
                ) : (
                  "Get Crop Recommendations"
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {recommendations && <CropRecommendations data={recommendations} />}
      </div>
    </TooltipProvider>
  )
}
