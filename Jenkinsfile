// Jenkins pipeline for mandi-automl.
//
// Build and test happen here; the actual deployment is delegated to the companion
// `mandi-infra` repository, which owns the Terraform and Ansible definitions.
//
// Every credential is referenced by its Jenkins credential ID. Nothing secret is
// written in this file.

pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 40, unit: 'MINUTES')
    }

    environment {
        REGISTRY     = 'ghcr.io'
        IMAGE_NAME   = 'ghcr.io/REPLACE_ME_OWNER/mandi-automl'
        IMAGE_TAG    = "${env.GIT_COMMIT ? env.GIT_COMMIT.take(7) : 'local'}"
        INFRA_REPO   = 'https://github.com/REPLACE_ME_OWNER/mandi-infra.git'
        UV_CACHE_DIR = "${env.WORKSPACE}/.uv-cache"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git rev-parse --short HEAD'
            }
        }

        stage('Lint & Test') {
            steps {
                sh '''
                    set -eu
                    curl -LsSf https://astral.sh/uv/install.sh | sh
                    export PATH="$HOME/.local/bin:$PATH"
                    uv sync --frozen --no-default-groups --group dev
                    uv run ruff check .
                    uv run ruff format --check .
                    uv run pytest -q
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh '''
                    set -eu
                    test -f models/index.json || {
                        echo "models/index.json is missing. Register models before building:"
                        echo "  uv run python -m automl.registry"
                        exit 1
                    }
                    test -f data/processed/weekly.parquet || {
                        echo "data/processed/weekly.parquet is missing. Build it first:"
                        echo "  uv run python -m automl.data.aggregate"
                        exit 1
                    }
                    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -t "${IMAGE_NAME}:latest" .
                '''
            }
        }

        stage('Smoke Test Image') {
            steps {
                sh '''
                    set -eu
                    docker rm -f mandi-smoke >/dev/null 2>&1 || true
                    docker run -d --name mandi-smoke -p 18000:8000 "${IMAGE_NAME}:${IMAGE_TAG}"
                    for attempt in $(seq 1 30); do
                        if curl -fsS http://127.0.0.1:18000/api/health; then break; fi
                        sleep 2
                    done
                    curl -fsS http://127.0.0.1:18000/api/health | grep -q '"status":"ok"'
                '''
            }
            post {
                always {
                    sh 'docker rm -f mandi-smoke >/dev/null 2>&1 || true'
                }
            }
        }

        stage('Push to GHCR') {
            when { branch 'main' }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'ghcr-credentials',
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_TOKEN'
                )]) {
                    sh '''
                        set -eu
                        echo "$GHCR_TOKEN" | docker login "${REGISTRY}" -u "$GHCR_USER" --password-stdin
                        docker push "${IMAGE_NAME}:${IMAGE_TAG}"
                        docker push "${IMAGE_NAME}:latest"
                        docker logout "${REGISTRY}"
                    '''
                }
            }
        }

        stage('Deploy') {
            when { branch 'main' }
            steps {
                dir('infra') {
                    git url: "${INFRA_REPO}", branch: 'main', credentialsId: 'github-credentials'
                }
                withCredentials([
                    sshUserPrivateKey(credentialsId: 'azure-vm-ssh', keyFileVariable: 'SSH_KEY'),
                    string(credentialsId: 'ansible-vault-password', variable: 'VAULT_PASS')
                ]) {
                    sh '''
                        set -eu
                        cd infra
                        echo "$VAULT_PASS" > .vault_pass
                        trap 'rm -f .vault_pass' EXIT
                        ansible-playbook ansible/deploy.yml \
                            --private-key "$SSH_KEY" \
                            --vault-password-file .vault_pass \
                            -e image_tag="${IMAGE_TAG}" \
                            -e image_name="${IMAGE_NAME}"
                    '''
                }
            }
        }

        stage('Verify Deployment') {
            when { branch 'main' }
            steps {
                withCredentials([string(credentialsId: 'app-base-url', variable: 'APP_URL')]) {
                    sh '''
                        set -eu
                        for attempt in $(seq 1 30); do
                            if curl -fsS "$APP_URL/api/health" | grep -q '"status":"ok"'; then
                                echo "deployment healthy"
                                exit 0
                            fi
                            sleep 5
                        done
                        echo "health check never returned ok"
                        exit 1
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker image prune -f --filter "until=72h" || true'
        }
        failure {
            echo "Build ${env.BUILD_NUMBER} failed at stage: ${env.STAGE_NAME}"
        }
    }
}
