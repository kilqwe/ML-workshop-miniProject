"use client";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface AreaChartProps {
  data: { 
    attribute: string; 
    PredictedPlayer: number;
    idealPlayer: number;
  }[];
}

export function ComparisonAreaChart({ data }: AreaChartProps) {
  return (
    <ResponsiveContainer width="100%" height={350}>
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <XAxis dataKey="attribute" />
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Legend />
        <Area type="monotone" dataKey="PredictedPlayer" stackId="1" stroke="#8884d8" fill="#8884d8" name="Your Player"/>
        <Area type="monotone" dataKey="idealPlayer" stackId="2" stroke="#82ca9d" fill="#82ca9d" name="Ideal Profile"/>
      </AreaChart>
    </ResponsiveContainer>
  );
}