import { SimpleClipperDashboard } from './components/SimpleClipperDashboard';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-[#09090b] dark:text-neutral-100 antialiased selection:bg-emerald-500/25 selection:text-emerald-950 dark:selection:bg-amber-500 dark:selection:text-black">
      <SimpleClipperDashboard />
    </div>
  );
}
