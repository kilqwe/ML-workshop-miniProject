"use client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, Cell } from 'recharts';

type LegendPayload = {
  value: string;
  type: string;
  color: string;
};

interface BarChartProps {
  data: { name: string; rating: number }[];
}

const COLORS = ["#3b82f6", "#22c55e", "#f97316", "#ef4444", "#8b5cf6"];

export function ComparisonBarChart({ data }: BarChartProps) {

  const legendPayload: Array<LegendPayload> = data.map((entry, index) => ({
    value: entry.name,
    type: 'square',
    color: COLORS[index % COLORS.length]
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[60, 100]} />
        <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
        <Tooltip />
        <Legend />
        <Bar dataKey="rating">
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}