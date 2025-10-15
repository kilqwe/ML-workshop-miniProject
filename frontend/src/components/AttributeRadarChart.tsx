"use client";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface RadarChartDataPoint {
  attribute: string; 
  predicted: number;
  match1: number;
  match2: number;
  match3: number;
}

interface RadarChartProps {
  data: RadarChartDataPoint[];
  names: {
    predicted: string;
    match1: string;
    match2: string;
    match3: string;
  };
}

export function AttributeRadarChart({ data, names }: RadarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <RadarChart cx="50%" cy="50%" outerRadius="75%" data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="attribute" />
        <Tooltip />
        <Legend />
        <Radar name={names.predicted} dataKey="predicted" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
        <Radar name={names.match1} dataKey="match1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.5} />
        <Radar name={names.match2} dataKey="match2" stroke="#f97316" fill="#f97316" fillOpacity={0.5} />
        <Radar name={names.match3} dataKey="match3" stroke="#ef4444" fill="#ef4444" fillOpacity={0.5} />
      </RadarChart>
    </ResponsiveContainer>
  );
}