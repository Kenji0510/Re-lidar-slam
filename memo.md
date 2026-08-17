# NAS
sudo mount -t nfs 10.10.10.10:/mnt/NAS/nfs /mnt/nas

# Mid70（既定値）
cargo run --release -- --lidar mid70

# Airy96
cargo run --release -- --lidar airy96

# ヘルプ
cargo run -- --help