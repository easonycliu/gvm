package gvm

import (
	"testing"
)

func TestParseGcgroupStat_Normal(t *testing.T) {
	content := `nr_submitted_kernels 12345
nr_ended_kernels 12340
nr_pending_kernels 5`

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 12345 {
		t.Errorf("NrSubmittedKernels: got %d, want 12345", stat.NrSubmittedKernels)
	}
	if stat.NrEndedKernels != 12340 {
		t.Errorf("NrEndedKernels: got %d, want 12340", stat.NrEndedKernels)
	}
	if stat.NrPendingKernels != 5 {
		t.Errorf("NrPendingKernels: got %d, want 5", stat.NrPendingKernels)
	}
}

func TestParseGcgroupStat_ZeroValues(t *testing.T) {
	content := `nr_submitted_kernels 0
nr_ended_kernels 0
nr_pending_kernels 0`

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 0 || stat.NrEndedKernels != 0 || stat.NrPendingKernels != 0 {
		t.Errorf("expected all zeros, got %+v", stat)
	}
}

func TestParseGcgroupStat_LargeValues(t *testing.T) {
	content := `nr_submitted_kernels 9999999999
nr_ended_kernels 9999999990
nr_pending_kernels 9`

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 9999999999 {
		t.Errorf("NrSubmittedKernels: got %d, want 9999999999", stat.NrSubmittedKernels)
	}
	if stat.NrPendingKernels != 9 {
		t.Errorf("NrPendingKernels: got %d, want 9", stat.NrPendingKernels)
	}
}

func TestParseGcgroupStat_Empty(t *testing.T) {
	stat, err := ParseGcgroupStat("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 0 || stat.NrEndedKernels != 0 || stat.NrPendingKernels != 0 {
		t.Errorf("expected all zeros for empty input, got %+v", stat)
	}
}

func TestParseGcgroupStat_ExtraWhitespace(t *testing.T) {
	content := `  nr_submitted_kernels   100  
  nr_ended_kernels   90  
  nr_pending_kernels   10  `

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 100 {
		t.Errorf("NrSubmittedKernels: got %d, want 100", stat.NrSubmittedKernels)
	}
	if stat.NrPendingKernels != 10 {
		t.Errorf("NrPendingKernels: got %d, want 10", stat.NrPendingKernels)
	}
}

func TestParseGcgroupStat_UnknownKeys(t *testing.T) {
	content := `nr_submitted_kernels 50
some_unknown_key 999
nr_ended_kernels 45
nr_pending_kernels 5`

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 50 || stat.NrEndedKernels != 45 || stat.NrPendingKernels != 5 {
		t.Errorf("unexpected values: %+v", stat)
	}
}

func TestParseGcgroupStat_MalformedLines(t *testing.T) {
	content := `nr_submitted_kernels 50
badline
nr_ended_kernels notanumber
nr_pending_kernels 5`

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 50 {
		t.Errorf("NrSubmittedKernels: got %d, want 50", stat.NrSubmittedKernels)
	}
	if stat.NrEndedKernels != 0 {
		t.Errorf("NrEndedKernels: got %d, want 0 (malformed)", stat.NrEndedKernels)
	}
	if stat.NrPendingKernels != 5 {
		t.Errorf("NrPendingKernels: got %d, want 5", stat.NrPendingKernels)
	}
}

func TestParseGcgroupStat_ColonFormat(t *testing.T) {
	content := `nr_submitted_kernels: 12345
nr_ended_kernels: 12340
nr_pending_kernels: 5`

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrSubmittedKernels != 12345 {
		t.Errorf("NrSubmittedKernels: got %d, want 12345", stat.NrSubmittedKernels)
	}
	if stat.NrEndedKernels != 12340 {
		t.Errorf("NrEndedKernels: got %d, want 12340", stat.NrEndedKernels)
	}
	if stat.NrPendingKernels != 5 {
		t.Errorf("NrPendingKernels: got %d, want 5", stat.NrPendingKernels)
	}
}

func TestParseGcgroupStat_PartialContent(t *testing.T) {
	content := `nr_pending_kernels 42`

	stat, err := ParseGcgroupStat(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat.NrPendingKernels != 42 {
		t.Errorf("NrPendingKernels: got %d, want 42", stat.NrPendingKernels)
	}
	if stat.NrSubmittedKernels != 0 || stat.NrEndedKernels != 0 {
		t.Errorf("missing keys should default to 0, got %+v", stat)
	}
}
