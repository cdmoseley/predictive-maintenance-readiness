# Fleet Maintenance Manual (Synthetic)

## Engine
Engine failure risk rises sharply after 4,000 operating hours without a major service.
Monitor oil analysis trends, coolant temperature, and fault codes EC-12 / EC-44.
If failures_last_90d >= 2 for the engine, schedule a teardown inspection within 14 days.

## Transmission
Transmission slip and delayed engagement often precede hard failures.
High fleet_same_component_failures_90d for transmissions signals a parts-quality or operating-condition issue —
cross-check unit training tempo and fluid change intervals.

## Brakes
Brake pad life is hours-driven. After 2,500 hours, inspect thickness every 30 days.
System failures in the last 30 days involving brakes require immediate deadline status until cleared.

## Electrical
Intermittent electrical faults are high false-positive drivers. Prefer trend confirmation
(multiple failures_last_90d) before deadline. Check battery SOC, ground straps, and CAN bus logs.

## Cooling
Cooling system failures cluster in high ambient temperature seasons.
If component_age_days > 900 and system_failures_last_30d > 0, flush and pressure-test before next mission.

## Hydraulics / Suspension
Hydraulic leaks reduce readiness faster than raw failure counts suggest.
Any active leak plus maintenance_actions_last_90d == 0 is a readiness red flag.
