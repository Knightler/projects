package config

import "testing"

func TestGetDSNUsesEnvWhenSet(t *testing.T) {
	t.Setenv("DB_DSN", "user:pass@tcp(localhost:3306)/custom")

	got := getDSN()

	if got != "user:pass@tcp(localhost:3306)/custom" {
		t.Fatalf("expected env DSN, got %q", got)
	}
}

func TestGetDSNUsesDefaultWhenEnvMissing(t *testing.T) {
	t.Setenv("DB_DSN", "")

	got := getDSN()

	if got != defaultDSN {
		t.Fatalf("expected default DSN, got %q", got)
	}
}
