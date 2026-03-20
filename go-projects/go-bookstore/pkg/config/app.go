package config

import (
	"os"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/mysql"
)

var db *gorm.DB

const defaultDSN = "root:@tcp(127.0.0.1:3306)/go_bookstore?charset=utf8mb4&parseTime=True&loc=Local"

func getDSN() string {
	dsn := os.Getenv("DB_DSN")
	if dsn == "" {
		return defaultDSN
	}
	return dsn
}

func Connect() {
	d, err := gorm.Open("mysql", getDSN())
	if err != nil {
		panic(err)
	}
	db = d
}

func GetDB() *gorm.DB {
	return db
}
