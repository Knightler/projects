package config

import (
	"os"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/mysql"
)

var db *gorm.DB

func getDSN() (string, bool) {
	dsn := os.Getenv("DB_DSN")
	if dsn == "" {
		return "", false
	}
	return dsn, true
}

func Connect() {
	dsn, ok := getDSN()
	if !ok {
		panic("DB_DSN environment variable is required")
	}

	d, err := gorm.Open("mysql", dsn)
	if err != nil {
		panic(err)
	}
	db = d
}

func GetDB() *gorm.DB {
	return db
}
