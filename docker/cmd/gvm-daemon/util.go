package main

import "time"

// ts returns a timestamp string for log messages.
func ts() string {
	return time.Now().Format("15:04:05")
}
