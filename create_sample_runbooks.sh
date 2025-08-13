#!/bin/bash
# Quick script to create sample SRE runbooks

mkdir -p data/runbooks

cat > data/runbooks/k8s_pod_crashloop.md << 'EOF'
# Kubernetes Pod CrashLoop Troubleshooting

## Symptoms
- Pods restarting repeatedly
- 5xx errors after deployment
- Application unavailable

## Diagnosis Steps
1. Check pod status: `kubectl get pods`
2. Examine pod logs: `kubectl logs <pod-name> --previous`
3. Describe pod events: `kubectl describe pod <pod-name>`
4. Check resource limits and requests
5. Verify health check configurations

## Resolution Actions
1. Review recent deployment changes
2. Check application startup time vs readiness probe timing
3. Verify environment variables and config maps
4. Scale down to single replica for debugging
5. Consider rollback to previous stable version

## Prevention
- Implement proper health checks
- Set appropriate resource limits
- Use staged deployments
- Monitor deployment metrics
EOF

cat > data/runbooks/disk_space.md << 'EOF'
# Disk Space Management

## Symptoms
- Write operations failing
- Disk space alerts at 95%
- Database errors related to storage

## Diagnosis Steps
1. Check disk usage: `df -h`
2. Find large files: `du -sh /* | sort -hr`
3. Check log rotation status
4. Identify temp files and old logs
5. Review database growth patterns

## Resolution Actions
1. Clean up log files: `find /var/log -name "*.log" -mtime +30 -delete`
2. Enable log rotation: `systemctl enable logrotate`
3. Archive old data
4. Expand disk if possible
5. Implement monitoring alerts

## Prevention
- Set up automated log rotation
- Monitor disk growth trends
- Implement log retention policies
- Set up proactive alerts at 80%
EOF

cat > data/runbooks/stale_config_cache.md << 'EOF'
# Stale Configuration Cache Issues

## Symptoms
- Feature flags not taking effect
- Old configuration values persisting
- Inconsistent application behavior

## Diagnosis Steps
1. Check config version: `curl /health/config-version`
2. Compare expected vs actual config
3. Verify cache invalidation mechanisms
4. Check for stuck background processes
5. Review config propagation logs

## Resolution Actions
1. Force cache refresh: `systemctl reload app-config`
2. Restart stateless services: `kubectl rollout restart deployment/app`
3. Purge cache manually if needed
4. Verify config distribution mechanism
5. Check for network connectivity issues

## Prevention
- Implement config version tracking
- Set up automated cache invalidation
- Add config validation steps
- Monitor config propagation delays
EOF

echo "✅ Created sample SRE runbooks in data/runbooks/"
ls -la data/runbooks/