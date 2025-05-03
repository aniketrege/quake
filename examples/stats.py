import json
import time

import docker

client = docker.from_env()


def log_docker_stats_sdk(log_file='docker_stats_sdk.log', duration=60, interval=5):
    end_time = time.time() + duration
    containers = client.containers.list()

    with open(log_file, 'w') as f:
        while time.time() < end_time:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    log_line = {
                        'timestamp': timestamp,
                        'id': container.id[:12],
                        'name': container.name,
                        'cpu_stats': stats.get('cpu_stats', {}),
                        'memory_stats': stats.get('memory_stats', {}),
                        'networks': stats.get('networks', {}),
                        'blkio_stats': stats.get('blkio_stats', {})
                    }
                    f.write(json.dumps(log_line) + '\n')
                except Exception as e:
                    f.write(json.dumps({'timestamp': timestamp, 'container': container.name, 'error': str(e)}) + '\n')
            time.sleep(interval)


log_docker_stats_sdk()
