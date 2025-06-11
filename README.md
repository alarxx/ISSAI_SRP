# ISSAI_SRP
> ISSAI Summer Research Project

---

Connect to remote computer:
```sh
ssh HPC_3070_8GB
```

----

# Connect with SSH to Github
https://docs.github.com/en/authentication

---

## Generating a new SSH-keys
```sh
ssh-keygen -t ed25519 -C "your_email@example.com"
```

## After generating SSH-keys
1. ssh-agent daemon:
```sh
eval "$(ssh-agent -s)"
```
>Agent pid 59566

2. Add ssh private key to daemon:
```sh
ssh-add ~/.ssh/id_ed25519
```

ssh-agent - это клиент для Linux, управляет private-key и отвечает за аутентификацию.

3. Adding a new SSH key to your GitHub account:
```shell
cat ~/.ssh/id_ed25519.pub
# Then select and copy the contents of the id_ed25519.pub file
# displayed in the terminal to your clipboard
```
публичный ключ даем гитхабу

---

## Testing your SSH connection

```sh
ssh -T git@github.com
# Attempts to ssh to GitHub
```
	-T      Disable pseudo-terminal allocation.
