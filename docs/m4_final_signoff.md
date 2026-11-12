# M4 sign-off checklist và proposed team message

**Trạng thái: PENDING TEAM SIGN-OFF. Không có approval được cấp trong package này.**
Protocol: `docs/m4_final_protocol.md`. Request: `docs/m4_final_signoff.request.json`.

## Approval checklist

- [ ] Ratify HGB native-categorical preprocessing, seed 42, M0/M3 whitelist,
      frozen seven families, original 15/31-leaf grid và validation-only raw pinball.
- [ ] Ratify fresh eligible train+validation refit, fixed test parameters,
      clip/sort correction, no calibration cho direct quantiles; report both variants.
- [ ] Acknowledge F1 đã exposed; xác minh F2–F4 exposure hoặc giữ pending/exploratory
      và không đưa confirmatory claims.
- [ ] Chốt P1/P4 calibration/refit hoặc ghi deferred_out_of_scope cho run này.
- [ ] Chốt paired block bootstrap (unit, block length, folds, repetitions, seed,
      pairing, multiple comparisons) hoặc defer và không claim CI/significance.
- [ ] Chốt bundle handoff hoặc exporter/schema cần làm sau; không giả định đã có
      `results/predictions/quantiles.csv` hay P1/P4 validation predictions.
- [ ] Chấp nhận retrospective weather-aware/sales proxy/source QA limitations;
      không test-guided policy selection, causal/operational/actual-profit claims.
- [ ] Reviewer xem exact snapshot và lưu người ký, UTC timestamp, evidence
      reference của sign-off; không điền người ký hoặc approval thay nhóm.

## Proposed message — chỉ là bản nháp, chưa gửi

> Nhờ nhóm review và sign-off M4 final quantile protocol trong
> `docs/m4_final_protocol.md` và exact snapshot ở `docs/m4_final_signoff.request.json`.
> Scope giữ nguyên F1–F4, M0/M3, tau 0.50/0.75/5/6, seed 42, HGB và original
> two-candidate grid; selection chỉ raw validation pinball, sau đó fresh refit.
> Không thay features/grid theo pilot test metrics. F1 đã exposed nên exploratory;
> F2–F4 confirmatory status còn chờ nhóm xác minh prior exposure.
> Xin chốt checklist preprocessing/grid, calibration/refit, bootstrap, handoff và
> claims; P1/P4/CI có thể defer explicit cho prediction-only run. Sau sign-off
> mới chạy 48 selection fits + 24 refits. Vui lòng ghi reviewer/time/evidence
> reference và resolutions trong approved copy. Chưa có final training nào chạy.

## Record và commands

Chuẩn bị request (exclusive create, **không training**):

```powershell
& '.\.venv\Scripts\python.exe' -B -m scripts.m4_final_run --prepare-signoff docs/m4_final_signoff.request.json
```

Nhóm review, tạo **copy mới** `docs/m4_final_signoff.approved.json`; không ghi đè
request. Chỉ khi có approval thật, điền:

- `approved: true`, `status: APPROVED_FOR_DIRECT_QUANTILE_EXECUTION`;
- `approved_by`, valid non-future UTC `approved_at_utc`, `approval_evidence_reference`;
- `f1_exposure_acknowledged: true`; F2–F4 status `pending`, `exploratory` hoặc
  `team_verified_uninspected` (trường hợp cuối bắt buộc có evidence reference);
- mỗi decision có status `approved` và resolution. Chỉ P1/P4 và bootstrap được
  phép `deferred_out_of_scope`, có resolution và giới hạn claims rõ ràng.

Không chỉnh snapshot/digest. Code/config/docs/input/package changes làm record
stale: tạo request M4 JSON tên mới và xin sign-off lại. Runner kiểm tra hashes,
nhưng không xác thực danh tính signer; evidence reference cần kiểm tra thủ công.

**Chỉ sau sign-off, command dưới sẽ thực sự training — không chạy trong lượt chuẩn bị:**

```powershell
& '.\.venv\Scripts\python.exe' -B -m scripts.m4_final_run --approval docs/m4_final_signoff.approved.json --execute
```

Runner khóa scope, tạo ID `final_f1_f4_m0_m3_<UTC>_<random>`; pilots không thể bị
overwrite. `approval_record.json` được copy byte-identical và hash-bind trong
audit trước fitting; input/code/record changes, preflight failures, count mismatch
và integrity failures đều dừng. New audit ở `results/experiments/m4_final_audit_*`;
model bundle vẫn do existing transactional writer quản lý. Audit checksum index
không tự chứa hash của chính nó. Không có approval JSON thật được tạo sẵn.
