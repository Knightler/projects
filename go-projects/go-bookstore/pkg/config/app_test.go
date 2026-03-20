package config

import "testing"

func TestGetDSNUsesEnvWhenSet(t *testing.T) {
	t.Setenv("DB_DSN", "user:pass@tcp(localhost:3306)/custom")

	got, ok := getDSN()

	if !ok {
		t.Fatal("expected DB_DSN to be present")
	}

	if got != "user:pass@tcp(localhost:3306)/custom" {
		t.Fatalf("expected env DSN, got %q", got)
	}
}

func TestGetDSNReturnsMissingWhenEnvMissing(t *testing.T) {
	t.Setenv("DB_DSN", "")

	got, ok := getDSN()

	if ok {
		t.Fatal("expected DB_DSN to be missing")
	}

	if got != "" {
		t.Fatalf("expected empty DSN when missing, got %q", got)
	}
}
