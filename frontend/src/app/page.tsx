"use client";
import { useState } from "react";
import { Loader2, Star, Users, Target, BarChart as BarChartIcon } from "lucide-react";
import {
  Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AttributeRadarChart } from "../components/AttributeRadarChart";
import { ComparisonBarChart } from "../components/ComparisonBarChart";
import { ComparisonAreaChart } from "../components/ComparisonAreaChart";

// --- Interfaces and Constants ---
interface SimilarPlayer {
  "Full Name": string;
  "Overall": number;
  "Best Position": string;
  "Club Name": string;
  [key: string]: any;
}
interface PredictionResult {
  predicted_exact_position: string;
  predicted_rating: number;
  predicted_group: string;
  similar_players: SimilarPlayer[];
  ideal_profile: { [key: string]: number } | null;
}

const outfieldFeatureInputs = [
  'Pace Total', 'Shooting Total', 'Passing Total',
  'Dribbling Total', 'Defending Total', 'Physicality Total'
];

const goalkeeperFeatureInputs = [
  'Goalkeeper Diving', 'Goalkeeper Handling', 'Goalkeeper Kicking',
  'Goalkeeper Positioning', 'Goalkeeper Reflexes'
];
const getInitialFeatures = (): { [key: string]: number } => {
  const allFeatures = [...outfieldFeatureInputs, ...goalkeeperFeatureInputs];
  return allFeatures.reduce((acc, feature) => {
    acc[feature] = 70;
    return acc;
  }, {} as { [key: string]: number });
};

// --- Helper Functions ---
const getRatingColor = (rating: number) => {
  if (rating >= 90) return "text-amber-400";
  if (rating >= 85) return "text-green-500";
  if (rating >= 75) return "text-blue-500";
  if (rating >= 65) return "text-gray-500";
  return "text-red-500";
};
const getRatingTier = (rating: number) => {
  if (rating >= 90) return "World Class";
  if (rating >= 85) return "Excellent";
  if (rating >= 75) return "Promising";
  if (rating >= 65) return "Decent";
  return "Developing";
};

// --- Component ---
export default function PlayerAttributeAnalyzer() {
  const [features, setFeatures] = useState<{ [key: string]: number }>(getInitialFeatures());
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("outfield");

  const handleFeatureChange = (name: string, value: number) => {
    const clampedValue = Math.max(40, Math.min(99, value));
    setFeatures({ ...features, [name]: clampedValue });
  };
  const handleInputChange = (name: string, value: string) => {
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue)) handleFeatureChange(name, numValue);
  };

  const fetchPrediction = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    const featuresToSend = activeTab === 'outfield'
      ? outfieldFeatureInputs.reduce((acc, key) => ({ ...acc, [key]: features[key] }), {})
      : goalkeeperFeatureInputs.reduce((acc, key) => ({ ...acc, [key]: features[key] }), {});

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(featuresToSend),
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction request failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderSliders = (featureList: string[]) => (
    featureList.map((featureName) => (
      <div key={featureName} className="space-y-2">
        <Label htmlFor={featureName} className="font-medium">{featureName}</Label>
        <div className="flex items-center gap-4">
          <Slider
            id={featureName} name={featureName} min={40} max={99} step={1}
            value={[features[featureName]]}
            onValueChange={(value) => handleFeatureChange(featureName, value[0])}
            className="flex-grow"
          />
          <Input
            type="number" min={40} max={99}
            value={features[featureName]}
            onChange={(e) => handleInputChange(featureName, e.target.value)}
            className="w-20 text-center"
          />
        </div>
      </div>
    ))
  );

  const radarChartData = result && result.similar_players.length >= 3
    ? (() => {
        const featureList = result.predicted_group === 'GK' ? goalkeeperFeatureInputs : outfieldFeatureInputs;
        const [match1, match2, match3] = result.similar_players;

        return featureList.map(feature => ({
          attribute: feature.replace(" Total", "").replace("Goalkeeper ","GK "),
          predicted: features[feature],
          match1: match1[feature] || 0,
          match2: match2[feature] || 0,
          match3: match3[feature] || 0,
        }));
      })()
    : [];
  
  const radarChartNames = result && result.similar_players.length >= 3
    ? {
        predicted: "Predicted Player",
        match1: result.similar_players[0]["Full Name"],
        match2: result.similar_players[1]["Full Name"],
        match3: result.similar_players[2]["Full Name"],
      }
    : { predicted: "Predicted Player", match1: "N/A", match2: "N/A", match3: "N/A" };

  const barChartData = result
    ? [
        { name: "Predicted Player", rating: result.predicted_rating },
        ...result.similar_players.map(p => ({ name: p["Full Name"], rating: p.Overall }))
      ]
    : [];

  const areaChartData = result && result.ideal_profile
    ? (result.predicted_group === 'GK' ? goalkeeperFeatureInputs : outfieldFeatureInputs).map(feature => ({
        attribute: feature.replace(" Total", "").replace("Goalkeeper ","GK "),
        PredictedPlayer: features[feature],
        idealPlayer: result.ideal_profile ? Math.round(result.ideal_profile[feature]) : 0,
      }))
    : [];

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900 font-sans">
      <div className="text-center py-16 px-6 bg-gray-900">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white">Football Player Potential Analyzer</h1>
        <p className="text-lg text-muted-foreground max-w-3xl mx-auto mt-4">
          Use our AI model to predict a player's rating and discover comparable real-world athletes.
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 max-w-7xl mx-auto gap-8 p-4 sm:p-6 lg:p-8">
        <Card className="w-full shadow-lg">
          <CardHeader>
            <CardTitle className="text-xl font-bold">Attribute Input</CardTitle>
            <CardDescription>Select a player type and adjust their skills.</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="outfield">Outfield Player</TabsTrigger>
                <TabsTrigger value="goalkeeper">Goalkeeper</TabsTrigger>
              </TabsList>
              <TabsContent value="outfield" className="space-y-6 pt-6">{renderSliders(outfieldFeatureInputs)}</TabsContent>
              <TabsContent value="goalkeeper" className="space-y-6 pt-6">{renderSliders(goalkeeperFeatureInputs)}</TabsContent>
            </Tabs>
          </CardContent>
          <CardFooter>
            <Button onClick={fetchPrediction} disabled={loading} className="w-full">
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {loading ? "Analyzing Potential..." : "Predict"}
            </Button>
          </CardFooter>
           {error && <p className="px-6 pb-4 text-sm font-medium text-destructive text-center">Error: {error}</p>}
        </Card>

        <Card className="w-full shadow-lg">
          <CardHeader>
            <CardTitle className="text-xl font-bold">Prediction Summary</CardTitle>
            <CardDescription>Primary results from our AI model.</CardDescription>
          </CardHeader>
          <CardContent className="min-h-[400px] flex flex-col justify-center">
            {loading ? (
              <div className="flex items-center justify-center h-full"><Loader2 className="h-8 w-8 animate-spin text-primary"/></div>
            ) : result ? (
              <div className="space-y-6 animate-fade-in w-full">
                <div className="text-center p-6 bg-muted rounded-lg">
                  <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground mb-2"><Star className="h-4 w-4" /> Predicted Overall Rating</div>
                  <h3 className={`text-7xl font-bold ${getRatingColor(result.predicted_rating)}`}>{Math.round(result.predicted_rating)}</h3>
                  <Badge className="mt-2">{getRatingTier(result.predicted_rating)}</Badge>
                </div>
                
                <div className="flex justify-around text-center">
                  <div>
                    <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground mb-2"><Target className="h-4 w-4" /> Position Group</div>
                    <Badge variant="secondary" className="text-lg">{result.predicted_group}</Badge>
                  </div>
                  <div>
                    <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground mb-2"><Target className="h-4 w-4" /> Exact Position</div>
                    <Badge variant="secondary" className="text-lg">{result.predicted_exact_position}</Badge>
                  </div>
                </div>

                <Separator />
                <div>
                  <h4 className="flex items-center justify-center gap-2 text-sm text-muted-foreground mb-3"><Users className="h-4 w-4" /> Top 3 Similar Players</h4>
                  <ul className="space-y-2">
                    {result.similar_players.slice(0, 3).map((player, index) => (
                      <li key={index} className="flex justify-between items-center bg-muted p-3 rounded-md">
                        <div>
                          <p className="font-medium text-sm">{player["Full Name"]}</p>
                          <p className="text-xs text-muted-foreground">
                            {player["Best Position"]} • {player["Club Name"]}
                          </p>
                        </div>
                        <Badge className={getRatingColor(player.Overall)} variant="outline">
                          {player.Overall} OVR
                        </Badge>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="text-center"><p className="text-muted-foreground">Your analysis results will appear here.</p></div>
            )}
          </CardContent>
        </Card>
      </div>

      {result && !loading && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 max-w-7xl mx-auto gap-8 px-4 sm:px-6 lg:px-8 pb-8 animate-fade-in">
            <Card className="w-full shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Users className="h-5 w-5" /> Player Profile Comparison</CardTitle>
                <CardDescription>Comparing your predicted player against the top 3 matches.</CardDescription>
              </CardHeader>
              <CardContent>
                {radarChartData.length > 0 ? (
                  <AttributeRadarChart data={radarChartData} names={radarChartNames} />
                ) : (
                  <p className="text-sm text-muted-foreground text-center">Not enough similar players found for a full comparison.</p>
                )}
              </CardContent>
            </Card>
            
            <Card className="w-full shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><BarChartIcon className="h-5 w-5" /> Rating Benchmark</CardTitle>
                <CardDescription>How your rating compares to the full list of similar pros.</CardDescription>
              </CardHeader>
              <CardContent>
                <ComparisonBarChart data={barChartData} />
              </CardContent>
            </Card>
          </div>

          {result.ideal_profile && (
            <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 pb-8 animate-fade-in">
                <Card className="w-full shadow-lg">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            Profile vs. Ideal {result.predicted_exact_position}
                        </CardTitle>
                        <CardDescription>How your player's attributes stack up against the average for their predicted position.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ComparisonAreaChart data={areaChartData} />
                    </CardContent>
                </Card>
            </div>
          )}
        </>
      )}
    </main>
  );
}