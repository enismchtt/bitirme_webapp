import { Calendar } from 'lucide-react'

export default function DateRangePicker({
  start, end, onStartChange, onEndChange, minDate, maxDate,
}) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <div>
        <label className="label flex items-center gap-1.5 mb-1.5">
          <Calendar size={12} /> Start date
        </label>
        <input
          type="date"
          value={start}
          min={minDate}
          max={maxDate}
          onChange={(e) => onStartChange(e.target.value)}
          className="input w-full"
        />
      </div>
      <div>
        <label className="label flex items-center gap-1.5 mb-1.5">
          <Calendar size={12} /> End date
        </label>
        <input
          type="date"
          value={end}
          min={start || minDate}
          max={maxDate}
          onChange={(e) => onEndChange(e.target.value)}
          className="input w-full"
        />
      </div>
    </div>
  )
}
