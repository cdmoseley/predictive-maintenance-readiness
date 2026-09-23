# Unit Readiness Policy (Synthetic)

## Readiness states
- GREEN: risk < 0.20 and no open deadline faults — fully mission capable
- AMBER: 0.20 <= risk < 0.50 or deferred non-critical maintenance — limited mission capable
- RED: risk >= 0.50 or safety-critical open fault — not mission capable

## Recommended actions
- GREEN: continue scheduled PM; no special action
- AMBER: schedule inspection within 7 days; brief commander on contingency
- RED: deadline vehicle; assign maintenance priority 1; notify readiness officer same day

## Operating philosophy
Missed failures cost more than false alarms. Prefer recall over precision at the decision threshold.
The predictive model is a decision-support signal, not an automatic deadline authority —
maintainers retain final say after reviewing contributing features and maintenance history.
