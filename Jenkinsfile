// Jenkins pipeline for mandi-automl.
//
// Build and test happen here; the actual deployment is delegated to the companion
// `devops_IA3` repository, which owns the Terraform and Ansible definitions.
//
// Every credential is referenced by its Jenkins credential ID. Nothing secret is
// written in this file.
//
// Jenkins runs locally (see docs/deployment.md), so a local install cannot receive
// a GitHub webhook. Poll SCM is used instead; the assignment allows either.

pipeline {
    agent any

    triggers {
        // Ask GitHub every two minutes whether main moved. `H` spreads the load so
        // several jobs on one Jenkins do not all poll on the same second.
        pollSCM('H/2 * * * *')
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 40, unit: 'MINUTES')
    }

    environment {
        REGISTRY     = 'ghcr.io'
        // GHCR rejects uppercase in a repository path, so this is not the literal
        // GitHub repo name (AutoML_IA3).
        IMAGE_NAME   = 'ghcr.io/dharshan-kumarj/automl_ia3'
        IMAGE_TAG    = "${env.GIT_COMMIT ? env.GIT_COMMIT.take(7) : 'local'}"
        INFRA_REPO   = 'https://github.com/dharshan-kumarj/Devops_IA3.git'
        INFRA_BRANCH = 'main'
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
                    # The image serves from models/ alone: the per-crop models and
                    # models/weekly_snapshot.csv, both committed. data/ is excluded by
                    # .dockerignore, so nothing under it is required to build here.
                    test -f models/index.json || {
                        echo "models/index.json is missing. Register models before building:"
                        echo "  uv run python -m automl.registry"
                        exit 1
                    }
                    test -f models/weekly_snapshot.csv || {
                        echo "models/weekly_snapshot.csv is missing. The API computes from it."
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
                    deleteDir()
                    git url: "${INFRA_REPO}", branch: "${INFRA_BRANCH}"
                }
                withCredentials([
                    sshUserPrivateKey(credentialsId: 'azure-vm-ssh', keyFileVariable: 'SSH_KEY'),
                    string(credentialsId: 'azure-vm-ip', variable: 'VM_IP'),
                    usernamePassword(
                        credentialsId: 'ghcr-credentials',
                        usernameVariable: 'GHCR_USER',
                        passwordVariable: 'GHCR_TOKEN'
                    )
                ]) {
                    sh '''
                        set -eu
                        cd infra/ansible
                        # The inventory is one host, so it is written here from the IP
                        # Terraform produced rather than committed to the infra repo.
                        trap 'rm -f inventory.ini' EXIT
                        {
                            echo "[app]"
                            echo "$VM_IP ansible_user=azureuser"
                        } > inventory.ini
                        ansible-playbook deploy.yml -i inventory.ini --private-key "$SSH_KEY" -e image_name="${IMAGE_NAME}" -e image_tag="${IMAGE_TAG}" -e ghcr_user="$GHCR_USER" -e ghcr_token="$GHCR_TOKEN"
                    '''
                }
            }
        }

        stage('Verify Deployment') {
            when { branch 'main' }
            steps {
                withCredentials([string(credentialsId: 'azure-vm-ip', variable: 'VM_IP')]) {
                    sh '''
                        set -eu
                        for attempt in $(seq 1 30); do
                            if curl -fsS "http://$VM_IP/api/health" | grep -q '"status":"ok"'; then
                                echo "deployment healthy: http://$VM_IP/"
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
        success {
            echo "Deployed ${env.IMAGE_NAME}:${env.IMAGE_TAG}"
        }
        failure {
            echo "Build ${env.BUILD_NUMBER} failed at stage: ${env.STAGE_NAME}"
        }
    }
}
