"""Validate virtual demo commands identically at both HTTP boundaries."""
import math


def parse_command(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected command object')
    command = payload.get('command')
    if command in ('pause', 'resume', 'restart'):
        return command
    if command == 'start_mode':
        mode=payload.get('mode')
        if mode not in ('load','preloaded'):raise ValueError('Invalid start mode')
        return {'command':'start_mode','mode':mode}
    if command == 'speed':
        speed = payload.get('speed')
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError('Speed must be numeric')
        if not math.isfinite(speed) or not .25 <= speed <= 16:
            raise ValueError('Speed must be between 0.25 and 16')
        return {'command': 'speed', 'speed': float(speed)}
    raise ValueError('Invalid command')
