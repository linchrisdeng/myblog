var w;

function setup() {
    createCanvas(640, 720);
    w = new Walker();
}

function draw() {
    background(50);
    w.update();
    w.display();
}

function Walker() {
    this.pos = createVector(width / 2, height / 2);
    this.vel = createVector(0, 0);

    this.update = function () {
        var mouse = createVector(mouseX, mouseY);
        this.acc = mouse.sub(this.pos);
        // this.acc = p5.Vector.fromAngle(0, PI * 2);
        // which give direction to the 180 degree 

        // this.acc.normalize();
        // this.acc.mult(5);
        // one line code
        this.acc.setMag(3);

        this.vel.add(this.acc);
        this.pos.add(this.vel);
    }

    this.display = function () {
        fill(255);
        ellipse(this.pos.x, this.pos.y, 48, 48);
    }
}