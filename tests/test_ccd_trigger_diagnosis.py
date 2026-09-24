from __future__ import annotations

import unittest

from devices.ccd_models import (
    ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST,
    ACQUISITION_EVENT_TRIGGER,
    ACQUISITION_EVENT_TRIGGER_IGNORED,
    FrameTriggerInput,
)
from devices.trigger_diagnosis import (
    ISOLATION_TEST,
    LIKELIHOOD_HIGH,
    LIKELIHOOD_LOW,
    LIKELIHOOD_MEDIUM,
    TriggerEvidence,
    describe_frame_trigger_input,
    diagnose_external_trigger,
    event_delta,
    frame_trigger_readbacks,
)

TTL_RISING = FrameTriggerInput(enabled=1, source=1, detection_raw=4, detection="RISING_EDGE", level_raw=1, level="LEVEL_TTL")
V24_RISING = FrameTriggerInput(enabled=1, source=2, detection_raw=4, detection="RISING_EDGE", level_raw=4, level="LEVEL_24VOLTS")


def evidence(**values) -> TriggerEvidence:
    defaults = dict(
        waits_for_trigger=True,
        triggered=False,
        encoder_delta=0,
        length_lines=100,
        compare_increment=2,
        trigger_input=TTL_RISING,
    )
    defaults.update(values)
    return TriggerEvidence(**defaults)


class FrameTriggerInputTextTests(unittest.TestCase):
    def test_known_values_are_explained_in_plain_words_and_short_codes(self):
        text = describe_frame_trigger_input(TTL_RISING)
        self.assertIn("代碼 1", text)
        self.assertIn("上升沿", text)
        self.assertIn("TTL（5V）", text)
        self.assertEqual(frame_trigger_readbacks(V24_RISING), {"FTS": "2", "FTD": "RISE", "FTL": "24V"})

    def test_unknown_numbers_stay_visible_and_unreadable_values_are_omitted(self):
        odd = FrameTriggerInput(source=3, detection_raw=64, level_raw=None)
        self.assertEqual(frame_trigger_readbacks(odd), {"FTS": "3", "FTD": "64"})
        self.assertIn("代碼 64", describe_frame_trigger_input(odd))
        self.assertIn("無法讀出", describe_frame_trigger_input(odd))
        self.assertIn("CamExpert", describe_frame_trigger_input(None))
        self.assertIn("CamExpert", describe_frame_trigger_input(FrameTriggerInput()))
        self.assertEqual(frame_trigger_readbacks(None), {})

    def test_disabled_frame_trigger_is_called_out(self):
        disabled = FrameTriggerInput(enabled=0, source=1, detection_raw=4, detection="RISING_EDGE")
        self.assertIn("未啟用", describe_frame_trigger_input(disabled))


class EventDeltaTests(unittest.TestCase):
    def test_counts_only_events_after_the_baseline(self):
        self.assertEqual(
            event_delta({ACQUISITION_EVENT_TRIGGER: 5, ACQUISITION_EVENT_TRIGGER_IGNORED: 2}, {ACQUISITION_EVENT_TRIGGER: 3}),
            {ACQUISITION_EVENT_TRIGGER: 2, ACQUISITION_EVENT_TRIGGER_IGNORED: 2},
        )
        self.assertEqual(event_delta({ACQUISITION_EVENT_TRIGGER: 1}, {ACQUISITION_EVENT_TRIGGER: 4}), {ACQUISITION_EVENT_TRIGGER: 0})

    def test_evidence_reads_backend_neutral_event_kinds(self):
        built = TriggerEvidence.from_events(
            {ACQUISITION_EVENT_TRIGGER: 1, ACQUISITION_EVENT_TRIGGER_IGNORED: 3, ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST: 2},
            waits_for_trigger=True,
            triggered=False,
            encoder_delta=10,
            length_lines=5,
            compare_increment=1,
        )
        self.assertEqual((built.trigger_events, built.ignored_events, built.line_trigger_too_fast), (1, 3, 2))
        self.assertEqual(built.expected_counts, 5)


class NoTriggerDiagnosisTests(unittest.TestCase):
    def test_waiting_until_two_lengths_have_passed(self):
        result = diagnose_external_trigger(evidence(encoder_delta=399))
        self.assertEqual((result.code, result.severity), ("waiting", "info"))
        self.assertFalse(result.is_problem)
        self.assertEqual(result.causes, ())

    def test_ttl_input_ranks_wiring_and_voltage_first_and_detection_last(self):
        result = diagnose_external_trigger(evidence(encoder_delta=400))

        self.assertEqual((result.code, result.severity), ("no_trigger", "error"))
        self.assertIn("Sensor 觸發", result.headline)
        self.assertIn("2.0 張", result.headline)
        likelihoods = [cause.likelihood for cause in result.causes]
        self.assertEqual(likelihoods, [LIKELIHOOD_HIGH, LIKELIHOOD_HIGH, LIKELIHOOD_MEDIUM, LIKELIHOOD_LOW])
        self.assertIn("輸入", result.causes[0].title)
        self.assertIn("TTL（5V）", result.causes[1].title)
        self.assertIn("24V", result.causes[1].action)
        self.assertIn("觸發方式", result.causes[-1].title)
        self.assertEqual(result.next_step, ISOLATION_TEST)
        self.assertTrue(any("被忽略 0 次" in fact for fact in result.facts))
        self.assertTrue(any("TTL（5V）" in fact for fact in result.facts))

    def test_24v_input_lowers_the_voltage_suspicion(self):
        result = diagnose_external_trigger(evidence(encoder_delta=400, trigger_input=V24_RISING))
        level = next(cause for cause in result.causes if "電壓" in cause.title)
        self.assertEqual(level.likelihood, LIKELIHOOD_MEDIUM)
        self.assertEqual(result.causes[0].likelihood, LIKELIHOOD_HIGH)

    def test_differential_input_is_flagged_as_unusable_for_a_plain_sensor(self):
        rs422 = FrameTriggerInput(source=1, detection_raw=4, detection="RISING_EDGE", level_raw=2, level="LEVEL_422")
        result = diagnose_external_trigger(evidence(encoder_delta=400, trigger_input=rs422))
        self.assertTrue(any("RS-422" in cause.title and cause.likelihood == LIKELIHOOD_HIGH for cause in result.causes))

    def test_unreadable_ccf_points_at_camexpert(self):
        result = diagnose_external_trigger(evidence(encoder_delta=400, trigger_input=None))
        self.assertTrue(all("CamExpert" in cause.action for cause in result.causes if cause.likelihood == LIKELIHOOD_HIGH))

    def test_grabber_without_trigger_events_is_never_blamed_for_a_missing_sensor(self):
        result = diagnose_external_trigger(evidence(encoder_delta=10_000, triggered=True, trigger_events_missing=True, frames=1))
        self.assertNotEqual(result.code, "no_trigger")
        self.assertTrue(any("不回報" in fact for fact in result.facts))

    def test_ignored_triggers_prove_the_sensor_reaches_the_grabber(self):
        result = diagnose_external_trigger(evidence(encoder_delta=50, ignored_events=3))
        self.assertEqual((result.code, result.severity), ("ignored", "error"))
        self.assertIn("3 次", result.headline)
        self.assertIn("上一張", result.causes[0].title)

    def test_completed_frames_while_waiting_read_as_healthy(self):
        result = diagnose_external_trigger(evidence(frames=2))
        self.assertEqual((result.code, result.severity), ("ok", "ok"))
        self.assertIn("2 張", result.headline)

    def test_notice_text_names_the_top_cause_and_the_panel(self):
        text = diagnose_external_trigger(evidence(encoder_delta=400)).notice_text()
        self.assertIn("最可能", text)
        self.assertIn("外部觸發診斷", text)


class LineAndTimingDiagnosisTests(unittest.TestCase):
    def test_progress_after_a_trigger(self):
        result = diagnose_external_trigger(evidence(triggered=True, trigger_events=1, encoder_delta=100))
        self.assertEqual(result.code, "running")
        self.assertIn("50 / 100", result.headline)
        self.assertIn("已收到 Sensor 觸發", result.headline)

    def test_frame_that_never_completes_blames_the_line_trigger(self):
        # 100 lines x 2 counts = 200; the no-frame threshold is 1.5 lengths plus one step.
        self.assertEqual(diagnose_external_trigger(evidence(triggered=True, encoder_delta=301)).code, "running")
        result = diagnose_external_trigger(evidence(triggered=True, encoder_delta=302))
        self.assertEqual((result.code, result.severity), ("no_line", "error"))
        self.assertIn("CMP_OUT", result.causes[0].title)
        self.assertIn("每 2 格", result.causes[1].why)

    def test_free_running_external_mode_does_not_wait_for_a_sensor(self):
        result = diagnose_external_trigger(evidence(waits_for_trigger=False, triggered=True, encoder_delta=60))
        self.assertEqual(result.code, "running")
        self.assertNotIn("Sensor", result.headline)
        self.assertFalse(any("Sensor 觸發" in fact for fact in result.facts))

    def test_timing_events_become_a_warning(self):
        result = diagnose_external_trigger(evidence(triggered=True, encoder_delta=20, line_trigger_too_fast=4))
        self.assertEqual((result.code, result.severity), ("timing", "warning"))
        self.assertIn("行觸發太快", result.causes[0].title)
        self.assertIn("每 2 格", result.causes[0].action)

    def test_reverse_counting_is_reported_first(self):
        result = diagnose_external_trigger(evidence(encoder_delta=-100, ignored_events=2))
        self.assertEqual(result.code, "reverse")
        self.assertIn("反向計數", result.causes[0].action)

    def test_full_text_lists_every_cause_with_why_and_action(self):
        result = diagnose_external_trigger(evidence(encoder_delta=400))
        text = result.text()
        self.assertTrue(text.startswith("[異常]"))
        self.assertEqual(text.count("為什麼："), len(result.causes))
        self.assertEqual(text.count("怎麼做："), len(result.causes))


if __name__ == "__main__":
    unittest.main()
